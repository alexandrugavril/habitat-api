import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class AvgLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, x, y):
        res = self.criterion(x, y)
        return res.mean(dim=1)


class RelativePositionPredictor(nn.Module):
    def __init__(self, cfg, visual_feat_size, target_encoding_size,
                 rnn_size):
        super().__init__()

        out_size = cfg.out_size

        self.loss_coeff = cfg.loss_coeff
        self.target = cfg.target

        self.net = nn.Sequential(
            nn.Linear(rnn_size, rnn_size),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_size, out_size),
        )

        self.criterion = nn.MSELoss()

    def forward(self, observations, prev_actions, masks,
                perception_embed, target_encoding, rnn_out):

        x = self.net(rnn_out)

        return x

    def set_per_element_loss(self):
        self.criterion = AvgLoss()

    def calc_loss(self, x, obs_batch, recurrent_hidden_states_batch,
                  prev_actions_batch, masks_batch, actions_batch):

        target = obs_batch[self.target]

        loss = self.criterion(x, target)
        return self.loss_coeff * loss



