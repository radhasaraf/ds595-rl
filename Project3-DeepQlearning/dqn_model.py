#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """
    Initialize a deep Q-learning network
    Architecture reference: Original paper for DQN
    (https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
    """
    def __init__(self, in_channels=4, num_actions=4):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Expected (sample) dummy input = zeros(batch_size, in_channels, 84, 84)
        h_out = w_out = self._conv2d_size_out(
            self._conv2d_size_out(self._conv2d_size_out(84, 8, 4), 4, 2),
            3,
            1
        )
        no_filters_last_conv_layer = 64

        self.in_features = int(h_out * w_out * no_filters_last_conv_layer)

        self.fc_stack = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    # Get the no. of features in the output of the conv-relu-layers-stack which
    # is required to be known for the Linear layer 'in_features' arg.

    # Following is simplified version. Checkout link below for the detailed one
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    @staticmethod
    def _conv2d_size_out(size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) / stride + 1

    # def forward(self, obs: Tensor) -> Tensor:
    #     """
    #     Passes an observation(state) through the network and generates action
    #     probabilities
    #     """
    #     obs = obs.to(device)
    #     intermediate_output = self.conv_relu_stack(obs)
    #     intermediate_output = intermediate_output.view(obs.size()[0], -1)
    #     return self.fc_stack(intermediate_output)

    def forward(self, obs: Tensor) -> Tensor:
        obs = obs.to(device)
        intermediate_output = self.conv_relu_stack(obs)
        intermediate_output = intermediate_output.view(obs.size(0), -1)
        values = self.value_stream(intermediate_output)
        advantages = self.advantage_stream(intermediate_output)
        qvals = values + (advantages - advantages.mean())

        return qvals