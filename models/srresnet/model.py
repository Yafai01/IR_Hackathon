import torch
import torch.nn as nn

from .blocks import ResidualBlock


class SRResNet(nn.Module):

    def __init__(
        self,
        num_blocks=16
    ):

        super().__init__()

        #############################################

        self.conv1 = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=9,
                padding=4
            ),

            nn.PReLU()

        )

        #############################################

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                ResidualBlock(64)
            )

        self.residuals = nn.Sequential(
            *blocks
        )

        #############################################

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64)

        )

        #############################################

        self.upsample = nn.Sequential(

            nn.Conv2d(
                64,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.PReLU()

        )

        #############################################

        self.output = nn.Conv2d(

            64,

            1,

            kernel_size=9,

            padding=4

        )

    def forward(self, x):

        x1 = self.conv1(x)

        x = self.residuals(x1)

        x = self.conv2(x)

        x = x + x1

        x = self.upsample(x)

        x = self.output(x)

        return x