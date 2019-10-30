#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import torch
import numpy as np

from habitat import Config, logger


class ObsExperienceRollout:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        all_num_steps,
        num_envs,
        observation_space,
        device=torch.device("cpu"),
    ):
        self.observations = {}

        assert all_num_steps % num_envs == 0, "Experience size not a " \
                                              "multiplier of num_envs"
        num_steps = all_num_steps // num_envs

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.ByteTensor(
                num_steps,
                num_envs,
                *observation_space.spaces[sensor].shape
            )
            self.observations[sensor].zero_()

        self.masks = torch.ByteTensor(num_steps, num_envs, 1)
        self.masks.fill_(1)

        self.all_num_steps = all_num_steps
        self.num_steps = num_steps
        self.device = device

        self.val_mask = torch.zeros_like(self.masks)
        self.has_validation = False
        self.val_pairs = []

        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        masks,
    ):
        step = self.step
        for sensor in observations:
            self.observations[sensor][step].copy_(
                observations[sensor]
            )

        self.masks[step].copy_(masks.unsqueeze(1))

        self.step = (self.step + 1) % self.num_steps

    def reset(self):
        for sensor in self.observations:
            self.observations[sensor].zero_()

        self.masks.fill_(1)
        self.val_mask.zero_()
        self.val_pairs = []
        self.has_validation = False
        self.step = 0

    def recurrent_generator(self, batch_size):
        num_steps = self.masks.size(0)
        num_recc = self.masks.size(1)
        num_envs = self.masks.size(2)

        idx_step = torch.arange(0, num_steps).view(-1, 1).expand(num_steps,
                                                                 num_envs).contiguous().view(-1)
        idx_envs = torch.arange(0, num_envs).view(1, -1).expand(num_steps,
                                                                num_envs).contiguous().view(-1)
        idxes = torch.stack([idx_step, idx_envs]).T

        perm = torch.randperm(len(idxes))

        for start_ind in range(0, len(perm), batch_size):
            observations_batch = defaultdict(list)
            masks_batch = []

            ind = idxes[perm[start_ind: start_ind + batch_size]]

            for iind in ind:
                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][iind[0], :, iind[1]]
                    )

                masks_batch.append(self.masks[iind[0], :, iind[1]])

            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 0
                ).view(batch_size, num_recc, *observations_batch[sensor].size()[3:])

            masks_batch = torch.stack(masks_batch, 0).view(batch_size, num_recc,
                                                           *masks_batch.size()[3:])
            # Return size (Batch_size X recurrence X obs size)
            yield (
                observations_batch,
                masks_batch,
            )

    def create_training_data(self, batch_size: int,
                             pos_dist: int, neg_dist: int,
                             val_split: float = 0.05):
        masks = self.masks
        obs = self.observations
        val_mask = self.val_mask
        val_pairs = self.val_pairs
        has_validation = self.has_validation

        num_steps = masks.size(0)
        num_envs = masks.size(1)

        idx_step = torch.arange(0, num_steps).\
            view(-1, 1).expand(num_steps, num_envs).contiguous().view(-1)
        idx_envs = torch.arange(0, num_envs).\
            view(1, -1).expand(num_steps, num_envs).contiguous().view(-1)

        idxes = torch.stack([idx_step, idx_envs]).T
        perm = torch.randperm(len(idxes))

        rr_steps = torch.zeros_like(masks).long()
        dones = 1 - masks

        for i in range(num_steps - 1)[::-1]:
            rr_steps[i] = (rr_steps[i + 1] + 1) * (masks[i + 1])

        idxes = idxes[perm]
        rr_steps = rr_steps.view(-1)[perm]
        positive = torch.randint(0, 2, (len(idxes), ))

        o1 = defaultdict(list)
        o2 = defaultdict(list)
        positive_label = []

        # Approximation of steps lost
        steps_lost = dones.sum() + dones.sum() * neg_dist * 0.5
        logger.info(f"[APPROXIMATION] Steps lost {steps_lost}")

        no_val_batches = int((len(idxes) - steps_lost) // batch_size *
                             val_split)
        no_val_batches = np.clip(no_val_batches, 1,
                                 len(idxes) // batch_size * 0.9)

        for (stp, ie), max_stp, pos in zip(idxes, rr_steps, positive):
            if has_validation:
                # Check if observation is in validation set
                if val_mask[stp, ie]:
                    continue

            if max_stp == 0:  # Last observation from episode
                continue

            # Define heuristic for positive pairs
            if pos:
                max_d = min(pos_dist, max_stp)
                pair_stp = torch.randint(stp,  stp + max_d + 1, (1,))
            elif max_stp >= neg_dist:
                pair_stp = torch.randint(stp + neg_dist,
                                         stp + max_stp + 1, (1,))
            else:
                continue

            for sensor in obs:
                o1[sensor].append(obs[sensor][stp, ie])
                o2[sensor].append(obs[sensor][pair_stp, ie])
            positive_label.append(pos)

            # Add observetion to validation
            if not has_validation:
                val_mask[stp, ie] = 1

            # Construct batch and return
            if len(positive_label) >= batch_size:
                for sensor in obs:
                    o1[sensor] = torch.stack(o1[sensor], 0)\
                        .view(batch_size, *obs[sensor].size()[2:])
                    o2[sensor] = torch.stack(o2[sensor], 0)\
                        .view(batch_size, *obs[sensor].size()[2:])
                positive_label = torch.stack(positive_label)

                if not has_validation:
                    if val_split > 0 and len(val_pairs) < no_val_batches:
                        val_pairs.append(
                            (o1, o2, positive_label)
                        )

                        # Filled validation set
                        if len(val_pairs) >= no_val_batches:
                            has_validation = True
                            self.has_validation = True
                        o1 = defaultdict(list)
                        o2 = defaultdict(list)
                        positive_label = []
                        continue

                yield (o1, o2, positive_label)

                o1 = defaultdict(list)
                o2 = defaultdict(list)
                positive_label = []

    def get_validation_pairs(self):
        assert self.has_validation, "No validation dataset allocated"
        for o1, o2, positive_label in self.val_pairs:
            yield (o1, o2, positive_label)

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, .., n ..), flatten it to size (t*n, ...).

        """
        x = tensor.transpose(1, 2)
        return x.view(t * n, *x.size()[2:])


if __name__ == "__main__":
    from argparse import Namespace

    num_steps = 12
    num_envs = 3
    observation_space = Namespace()
    observation_space.spaces = {"rgb": torch.rand(3, 5, 5)}
    device = torch.device("cpu")

    masks = torch.zeros(num_steps, num_envs).long()
    obs = {"rgb": torch.rand(num_steps, num_envs, 3, 5, 5)}
    pos_dist = 3
    neg_dist = 5
    batch_size = 4

    rollout = ObsExperienceRollout(num_steps, num_envs, observation_space,
                                   device=device)
