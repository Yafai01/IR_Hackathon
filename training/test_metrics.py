import numpy as np

from utils.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    calculate_rmse
)

target = np.random.rand(512,512)

prediction = target + np.random.normal(
    0,
    0.02,
    target.shape
)

prediction = np.clip(prediction,0,1)

print()

print("PSNR :", calculate_psnr(prediction,target))

print("SSIM :", calculate_ssim(prediction,target))

print("MAE  :", calculate_mae(prediction,target))

print("RMSE :", calculate_rmse(prediction,target))