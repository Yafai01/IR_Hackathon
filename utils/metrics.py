import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calculate_psnr(pred, target):

    return peak_signal_noise_ratio(
        target,
        pred,
        data_range=1.0
    )


def calculate_ssim(pred, target):

    return structural_similarity(
        target,
        pred,
        data_range=1.0
    )


def calculate_mae(pred, target):

    return np.mean(np.abs(pred - target))


def calculate_rmse(pred, target):

    return np.sqrt(np.mean((pred - target) ** 2))