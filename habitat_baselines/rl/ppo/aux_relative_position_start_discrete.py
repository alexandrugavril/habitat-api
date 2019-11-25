import torch
import torch.nn as nn


class RelativeDiscreteStartPositionPredictor(nn.Module):
    def __init__(self, cfg, visual_feat_size, target_encoding_size,
                 rnn_size):
        super().__init__()

        without_rotation = 1
        self.out_size = out_size = cfg.out_size - without_rotation

        self.loss_coeff = cfg.loss_coeff
        self.target = cfg.target
        self.max_value = max_value = cfg.max_value
        # self.discrete = cfg.discrete

        self.x = nn.Sequential(
            nn.Linear(rnn_size, rnn_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(rnn_size, max_value*2),
        )
        self.y = nn.Sequential(
            nn.Linear(rnn_size, rnn_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(rnn_size, max_value*2),
        )


        self.criterion = nn.CrossEntropyLoss()

    def forward(self, observations, prev_actions, masks,
                perception_embed, target_encoding, rnn_out):

        x = self.x(rnn_out)
        y = self.y(rnn_out)

        return torch.cat([x,y], dim=1)

    def calc_loss(self, x, obs_batch, recurrent_hidden_states_batch,
                  prev_actions_batch, masks_batch, actions_batch):

        target = obs_batch[self.target]
        target = target[:, :self.out_size]
        max_v = self.max_value
        target = (target + max_v).long()
        loss = self.criterion(x[:, :max_v*2], target[:, 0])
        loss += self.criterion(x[:, max_v*2:], target[:, 1])
        return self.loss_coeff * loss



