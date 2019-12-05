import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy


class ActionPrediction(nn.Module):
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

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, observations, prev_actions, masks,
                perception_embed, target_encoding, rnn_out):

        x = self.net(rnn_out)

        return x

    def set_per_element_loss(self):
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def calc_loss(self, x, obs_batch, recurrent_hidden_states_batch,
                  prev_actions_batch, masks_batch, actions_batch):

        target = prev_actions_batch.squeeze(1)

        loss = self.criterion(x, target)
        return self.loss_coeff * loss



