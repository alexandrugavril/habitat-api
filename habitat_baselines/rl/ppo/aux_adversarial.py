import torch
import torch.nn as nn
import numpy as np
import collections
import itertools

from habitat_baselines.common.rollout_storage_experience import ObsExperienceMemory
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class


class Discriminator(nn.Module):
    def __init__(self, in_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_size, in_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_size // 2, 2)
        )

    def forward(self, embeddings):
        x = self.net(embeddings)
        return x


class AdversarialDomainAdaptation(nn.Module):
    def __init__(self, cfg, visual_feat_size, target_encoding_size,
                 rnn_size, **kwargs):
        super().__init__()

        self.observation_space = kwargs['observation_space']
        self.loss_coeff = cfg.loss_coeff
        self.env_memory_size = int(cfg.env_memory_size)
        self.real_mem_size = int(cfg.real_memory_size)
        self._finetune_rl = cfg.finetune_rl
        self.train_epochs = cfg.train_epochs
        self.batch_size = cfg.batch_size
        self.optim_name_v = cfg.optim_name_v
        self.optim_param_v = cfg.optim_param_v
        self.optim_step_d = cfg.optim_step_d
        self.optim_step_v = cfg.optim_step_v
        optim_name_d = cfg.optim_name_d
        optim_param_d = cfg.optim_param_d

        self.log_freq = cfg.log_freq

        feature_loss = cfg.feature_loss

        self.master = True

        spaces = self.observation_space.spaces

        data_sizes = [
            torch.Size(spaces["rgb"].shape),
            torch.Size(spaces["depth"].shape),
            torch.Size([visual_feat_size])
        ]
        self.memory_env = ObsExperienceMemory(self.env_memory_size,
                                              data_sizes)
        self.memory_real = ObsExperienceMemory(self.real_mem_size,
                                               data_sizes[:-1])

        # Load discriminator network stuff
        self.discriminator = Discriminator(visual_feat_size)
        self.d_criterion = nn.CrossEntropyLoss()

        optimizer = getattr(torch.optim, optim_name_d)
        self.d_optimizer = optimizer(
            self.discriminator.parameters(),
            **dict(optim_param_d)
        )

        self.v_optimizer = None

        self.v_feature_criterion = getattr(torch.nn.functional, feature_loss)

        self.parent = None
        self.visual_encoder = None

    def _load_real_data(self, config):
        real_mem = self.memory_real

        config = config.clone()
        config.defrost()
        config.TASK_CONFIG.DATASET.TYPE = "PepperRobot"
        config.ENV_NAME = "PepperPlaybackEnv"
        config.NUM_PROCESSES = 1
        config.freeze()

        real_env = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )

        observations = real_env.reset()
        real_mem.insert([observations[0]["rgb"], observations[0]["depth"]])

        print(f"Loading {self.real_mem_size}  ROBOT observations ...")
        for i in range(self.real_mem_size):
            outputs = real_env.step([0])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            real_mem.insert([observations[0]["rgb"], observations[0]["depth"]])
        print(f"Loaded {self.real_mem_size} observations from ROBOT dataset")

    def set_trainer(self, parent):
        self.parent = parent
        self.visual_encoder = self.parent.actor_critic.net.visual_encoder

        optimizer = getattr(torch.optim, self.optim_name_v)
        self.v_optimizer = optimizer(
            self.visual_encoder.parameters(),
            **dict(self.optim_param_v)
        )

        self._load_real_data(parent.config)
        # Disable PPO training to collect env memory
        self.parent._train = False

    def forward(self, observations, prev_actions, masks,
                perception_embed, target_encoding, rnn_out):
        batch_size = len(perception_embed)

        out = torch.Tensor(batch_size, 0)
        out = out.to(perception_embed.device)
        for i in range(batch_size):
            self.memory_env.insert([
                observations["rgb"][i],
                observations["depth"][i],
                perception_embed[i]
            ])

            if self.memory_env.filled_memory_size >= self.real_mem_size:
                print("Gathered env memory")
                self.train_adaptation(self.train_epochs)
                break

        return out

    def train_adaptation(self, epochs):
        trainer = self.parent
        env_memory = self.memory_env
        real_memory = self.memory_real
        batch_size = self.batch_size
        log_freq = self.log_freq
        device = next(self.visual_encoder.parameters()).device

        num_batches_env = env_memory.calc_num_batches(batch_size)
        num_batches_real = real_memory.calc_num_batches(batch_size)

        print(f"Train adaptation, for {epochs} epochs;\n\t"
              f"Num batches from env: {num_batches_env}\n\t"
              f"Num batches from real: {num_batches_real}\n\t")

        count_checkpoints = 0
        trainer.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
        count_checkpoints += 1

        avg_len = 30
        accs = {
            name: collections.deque(maxlen=avg_len)
            for name in ["loss_d", "loss_adv", "loss_feat", "d_acc"]
        }
        batch_count = 0

        optim_step_d = self.optim_step_d
        optim_step_v = self.optim_step_v

        optim_all = optim_step_d + optim_step_v
        with torch.enable_grad():
            for ep, batch_idx in itertools.product(range(epochs),
                                                   range(num_batches_env)):

                env_rgb, env_depth, gt_env_features = env_memory.sample(
                    batch_size, device=device)
                real_rgb, real_depth = real_memory.sample(batch_size,
                                                          device=device)

                env_obs = {"rgb": env_rgb, "depth": env_depth}
                real_obs = {"rgb": real_rgb, "depth": real_depth}

                real_feat = self.visual_encoder(real_obs)
                env_feat = self.visual_encoder(env_obs)

                features = torch.cat([real_feat.detach(),
                                      env_feat.detach()], dim=0)

                if batch_count % optim_all < optim_step_d:
                    d_out = self.discriminator(features)
                    d_targets = torch.zeros(len(features),
                                            device=features.device).long()
                    d_targets[len(env_feat):] = 1

                    acc = (d_out.max(dim=1)[1] == d_targets).sum()
                    accs["d_acc"].append(acc.item() / len(d_out))

                    # Discriminator loss
                    loss_d = self.d_criterion(d_out, d_targets)
                    accs["loss_d"].append(loss_d.item())

                    self.d_optimizer.zero_grad()
                    loss_d.backward()
                    self.d_optimizer.step()
                else:
                    # Visual encoder adaptation
                    r_out = self.discriminator(real_feat)
                    r_targets = torch.ones(len(r_out),
                                           device=features.device).long()

                    loss_adv = self.d_criterion(r_out, r_targets)
                    loss_feat = self.v_feature_criterion(env_feat,
                                                         gt_env_features)
                    loss_v = loss_adv + loss_feat * 10
                    accs["loss_adv"].append(loss_adv.item())
                    accs["loss_feat"].append(loss_feat.item())

                    self.v_optimizer.zero_grad()
                    loss_v.backward()
                    self.v_optimizer.step()

                batch_count += 1
                if batch_count % log_freq == 0:
                    print_list = [(k, np.mean(v)) for k, v in accs.items()]
                    print(f"Loss: {print_list} ")
                    trainer.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

    def set_per_element_loss(self):
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def calc_loss(self, x, obs_batch, recurrent_hidden_states_batch,
                  prev_actions_batch, masks_batch, actions_batch):

        target = prev_actions_batch.squeeze(1)

        loss = self.criterion(x, target)
        return self.loss_coeff * loss
