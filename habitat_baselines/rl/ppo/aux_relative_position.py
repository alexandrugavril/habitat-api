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
                 rnn_size, **kwargs):
        super().__init__()

        out_size = cfg.out_size

        self.loss_coeff = cfg.loss_coeff
        self.target = cfg.target

        self.net = nn.Sequential(
            nn.Linear(rnn_size, rnn_size),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_size, out_size),
        )

        # Split heads
        # self.preprocess_net = nn.Sequential(
        #         nn.Linear(rnn_size, rnn_size),
        #         nn.LeakyReLU(inplace=True),
        #         nn.Linear(rnn_size, rnn_size),
        #         nn.LeakyReLU(inplace=True),
        # )
        # self.net = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(rnn_size, 1),
        #         # nn.ReLU(inplace=True),
        #         # nn.Linear(rnn_size, rnn_size // 2),
        #         # nn.ReLU(inplace=True),
        #         # nn.Linear(rnn_size // 2, 1),
        #     ) for _ in range(out_size)
        # ])

        self.criterion = nn.MSELoss()

    def forward(self, observations, prev_actions, masks,
                perception_embed, target_encoding, rnn_out):

        # Split heads
        # x = self.preprocess_net(rnn_out)
        # x = [c_net(x) for c_net in self.net]
        # return torch.cat(x, dim=1)

        x = self.net(rnn_out)
        return x

    def set_per_element_loss(self):
        self.criterion = AvgLoss()

    def calc_loss(self, x, obs_batch, recurrent_hidden_states_batch,
                  prev_actions_batch, masks_batch, actions_batch):

        target = obs_batch[self.target]

        loss = self.criterion(x, target)

        # dist = torch.sqrt((x-target)**2)
        # for i in range(len(target)):
        #     print(target[i], x[i], "(", dist[i], ")")

        return self.loss_coeff * loss



