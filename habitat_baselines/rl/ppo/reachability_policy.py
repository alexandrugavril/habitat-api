
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from habitat import Config, logger

from habitat_baselines.rl.models.reachability_net import ReachabilityFeatures
from habitat_baselines.rl.models.reachability_net import ReachabilityNet

from habitat_baselines.common.rollout_storage_experience import ObsExperienceRollout
from habitat_baselines.common.tensorboard_utils import TensorboardWriter


class ReachabilityPolicy(nn.Module):

    def __init__(self, cfg: Config, num_envs, observation_spaces, device,
                 with_training=True, tb_dir=None):
        super().__init__()

        # Get config param:
        self.experience_buffer_size = cfg.experience_buffer_size // num_envs
        self.with_training = with_training
        self.similarity_aggregation = cfg.similarity_aggregation
        self.similarity_percentile = cfg.similarity_percentile
        self.curiosity_bonus_scale_a = cfg.curiosity_bonus_scale_a
        self.reward_shift_b = cfg.reward_shift_b
        self.novelty_threshold = cfg.novelty_threshold
        self.num_train_epochs = cfg.num_train_epochs
        self.batch_size = cfg.batch_size
        self.max_action_distance_k = cfg.max_action_distance_k
        self.negative_sample_multiplier = cfg.negative_sample_multiplier
        self.log_freq = cfg.log_freq

        self.device = device

        # Initialize training experience memory
        if with_training:
            self.rollout = ObsExperienceRollout(cfg.experience_buffer_size,
                                                num_envs,
                                                observation_spaces,
                                                cfg.num_recurrent_steps)

        # Initialize reachability feature extractor & network
        self.rex = ReachabilityFeatures(self.envs.observation_spaces[0],
                                        pretrained=True)

        self.r_net = ReachabilityNet(cfg.feature_extractor_size)

        # Initialize memory
        self._memory = torch.FloatTensor(num_envs,
                                         cfg.memory_size,
                                         cfg.feature_extractor_size).to(device)
        self._memory_mask = torch.LongTensor(num_envs)

        # Initialize training params
        if with_training:
            optimizer = getattr(torch.optim, cfg.optimizer)
            self.optimizer = optimizer(
                list(self.rex.parameters()) + list(self.r_net.parameters()),
                **cfg.optimizer_args.__dict__
            )
            self.criterion = nn.CrossEntropyLoss()

        self._step = 0

        assert tb_dir is not None, "No tensorboard directory set"
        self.tb_writer = TensorboardWriter(tb_dir, flush_secs=30)

    def act(self, batch: dict, actions: torch.tensor, rewards: torch.tensor,
            masks: torch.tensor):
        self.eval()

        a = self.curiosity_bonus_scale_a
        b = self.reward_shift_b

        # Add to training memory
        if self.training and self.with_training:
            self.rollout.insert(batch, masks)

        # Get similarity scores compared with memory
        similarity_scores, r_features = self.similarity_to_memory(batch)

        # Calculate intrinsic reward
        ir = a * (b - similarity_scores)

        # Check if we add this new feature to memory
        self.add_to_memory(r_features, similarity_scores)

        self._step += 1

        if self._step % self.experience_buffer_size == 0:
            self.train_step()

        rewards.add_(ir)
        return ir

    def add_to_memory(self, obs_features, similarities):
        """
            Check if new obs feature is added to memory
        """
        mem = self._memory
        mem_mask = self._memory_mask
        novelty_threshold = self.novelty_threshold
        device = self.device

        mem_size = mem.size(1)
        no_envs = mem.size(0)

        add_mask = similarities > novelty_threshold

        full_mem = mem_size == mem_mask

        # Select random index if memory is full or last empty idx
        rand_idx = torch.randint(mem_size, (no_envs,), device=device)
        # increase non empty idx
        mem_mask[~full_mem & add_mask] += 1
        add_idx = full_mem * rand_idx + (~full_mem) * (mem_mask - 1)

        sel_idx = add_mask.nonzero().squeeze(1)
        indices = add_idx[add_mask]
        if len(sel_idx) > 0:
            sel = torch.stack([sel_idx, indices]).T
            for i, j in sel:
                mem[i, j].copy_(obs_features[i])

    def train_step(self):
        no_epochs = self.num_train_epochs
        batch_size = self.batch_size
        positive_dist = self.max_action_distance_k
        negative_dist = positive_dist * self.negative_sample_multiplier

        rex = self.rex
        r_net = self.r_net
        optimizer = self.optimizer
        criterion = self.criterion
        log_freq = float(self.log_freq)
        writer = self.tb_writer

        train_step = 0
        for epoch in range(no_epochs):

            train_loss = 0
            num_batches = 0
            # Train
            self.train()
            dataset = self.rollout.create_training_data(batch_size,
                                                        positive_dist,
                                                        negative_dist)
            for i, (o1, o2, positive_label) in enumerate(dataset):
                o1_feat = rex(o1)
                o2_feat = rex(o2)

                r_scores = r_net(o1_feat, o2_feat)

                loss = criterion(r_scores, positive_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                if (i + 1) % log_freq == 0:
                    train_loss = train_loss / log_freq
                    logger.info(
                        f"[TRAIN] Epoch: {epoch} | iter {i} | "
                        f"loss: {train_loss}"
                    )
                    writer.add_scalar("train_loss", train_loss, train_step)
                    train_loss = 0

                train_step += 1
                num_batches += 1

            # Validate
            self.eval()
            val_loss = []
            val_acc = 0
            num_items = 0
            dataset = self.rollout.get_validation_pairs()
            for i, (o1, o2, positive_label) in enumerate(dataset):
                o1_feat = rex(o1)
                o2_feat = rex(o2)

                r_scores = r_net(o1_feat, o2_feat)

                loss = criterion(r_scores, positive_label)
                val_loss.append(loss.item())

                acc = (r_scores.max(dim=1)[1] == positive_label).sum().item()
                val_acc += acc
                num_items += len(positive_label)

            val_loss = np.mean(val_loss)
            mean_acc = acc / float(num_items)
            logger.info(
                f"[VAL] Epoch: {epoch} | loss {val_loss} |"
                f" ACC : {mean_acc}"
                f" (out of {num_items} items"
            )
            writer.add_scalar("val_loss", val_loss, train_step)
            writer.add_scalar("val_acc", mean_acc, train_step)

            logger.info(f"Num batches: train {num_batches} /"
                        f" eval {num_items// batch_size}")

    def similarity_to_memory(self, obs: dict):
        mem = self._memory
        mem_mask = self._memory_mask
        obs_features = self.rex(obs)

        similarities = torch.zeros(obs_features.size(0), device=self.device)

        feat_size = obs_features.size(1)

        crt_features = []
        mem_features = []
        for ienv in range(mem.size(0)):
            crt_features.append(obs_features[ienv].unsqueeze(0).expand(
                mem_mask[ienv], feat_size
            ))
            mem_features.append(mem[ienv, :mem_mask])

        crt_features = torch.cat(crt_features, dim=0)
        mem_features = torch.cat(mem_features, dim=0)

        r_scores = self.r_net(crt_features, mem_features)  # Without softmax
        r_scores = F.softmax(r_scores)
        r_scores = r_scores[:, 1]  # 0 for negative 1 for positive

        st_idx = 0
        for ienv, end_idx in enumerate(mem_mask.cumsum(0)):
            env_scores = r_scores[st_idx: end_idx]
            env_score = self.similarity_score(env_scores.cpu().numpy())
            similarities[ienv] = env_score

            st_idx = end_idx

        return similarities, obs_features

    def similarity_score(self, similarities: np.ndarray):
        similarity_aggregation = self.similarity_aggregation
        aggregated = None

        if similarity_aggregation == 'max':
            aggregated = np.max(similarities)
        elif similarity_aggregation == 'nth_largest':
            n = min(10, len(similarities))
            aggregated = np.partition(similarities, -n)[-n]
        elif similarity_aggregation == 'percentile':
            percentile = self.similarity_percentile
            aggregated = np.percentile(similarities, percentile)
        elif similarity_aggregation == 'relative_count':
            # Number of samples in the memory similar to the input observation.
            count = sum(similarities > 0.5)
            aggregated = float(count) / len(similarities)

        return aggregated

    def forward(self, *x):
        raise NotImplementedError


if __name__ == "__main__":
    import torch

    # test add
    no_envs = 3
    num_feat = 15
    novelty_threshold = 0.5
    mem = torch.zeros(no_envs, 10, num_feat).cuda()
    similarities = torch.rand(no_envs).cuda()
    obs_features = torch.rand(no_envs, num_feat).cuda()
    mem_mask = torch.randint(10, (no_envs,)).cuda()
    device = torch.device("cuda")
