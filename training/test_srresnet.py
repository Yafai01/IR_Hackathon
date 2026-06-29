import os
import sys

sys.path.append(os.path.abspath("."))

import torch

from models.srresnet.model import SRResNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SRResNet().to(device)

x = torch.randn(2, 1, 512, 512).to(device)

with torch.no_grad():
    y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)