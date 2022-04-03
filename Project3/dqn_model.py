#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn, Tensor, zeros


class DQN(nn.Module):
    """
    Initialize a deep Q-learning network
    Architecture reference: Original paper for DQN
    (https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
    """
    def __init__(self, in_channels=4, num_actions=4, batch_size=32):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        dummy_input = zeros(self.batch_size, in_channels, 84, 84)

        self.fc_stack = nn.Sequential(
            nn.Linear(self._get_conv2d_output_features(dummy_input), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def _get_conv2d_output_features(self, dummy_input) -> int:
        """
        Get the no of features in the output of the conv-relu-layers-stack which
        is required to be known for the Linear layer 'in_features' arg.
        """
        dummy_conv_output = self.conv_relu_stack(dummy_input)
        dummy_conv_output = dummy_conv_output.view(self.batch_size, -1)
        return dummy_conv_output.shape[1]  # 64*7*7

    def forward(self, obs: Tensor) -> Tensor:
        """
        Passes an observation(state) through the network and generates action
        probabilities
        """
        intermediate_output = self.conv_relu_stack(obs)
        intermediate_output = intermediate_output.view(obs.size()[0], -1)
        return self.fc_stack(intermediate_output)
