import os
import sys

sys.path.append(os.path.abspath("."))

import torch

from models.colorizer.model import ColorizationUNet

model = ColorizationUNet()

x = torch.randn(2, 1, 1024, 1024)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)