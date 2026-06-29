import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Standard Residual Block
    """

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(channels),

            nn.PReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(channels)

        )

    def forward(self, x):

        return x + self.block(x)