import torch
import torch.nn as nn

from .blocks import DoubleConv
from .blocks import Down
from .blocks import Up


class UNetSR(nn.Module):

    def __init__(self):

        super().__init__()

        self.input = DoubleConv(1, 64)

        self.down1 = Down(64, 128)

        self.down2 = Down(128, 256)

        self.down3 = Down(256, 512)

        self.up1 = Up(512, 256)

        self.up2 = Up(256, 128)

        self.up3 = Up(128, 64)

        self.final = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        x1 = self.input(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x = self.up1(x4, x3)

        x = self.up2(x, x2)

        x = self.up3(x, x1)

        x = self.final(x)

        return x