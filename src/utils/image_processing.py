import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
from scipy.ndimage import rotate
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

def apply_noise_brightness_contrast(
    x: torch.Tensor,
    noise_std: float = 0.02,       # Gaussian noise std in [0,1] intensity units
    brightness: float = 0.0,       # add this offset (e.g., +0.05 brighter, -0.05 darker)
    contrast: float = 0.8,         # <1 lowers contrast, 1=no change, >1 increases
    per_channel_mean: bool = False # contrast pivot: False=global per-image mean, True=per-channel
) -> torch.Tensor:
    """
    img should be float in [0,1]. Returns transformed tensor clamped to [0,1].
    """

    # Ensure floating point
    if not x.is_floating_point():
        x = x.float()

    # Contrast around mean
    if per_channel_mean:
        mean = x.mean(dim=(-2, -1), keepdim=True)        # [N,C,1,1]
    else:
        mean = x.mean(dim=(1,2,3), keepdim=True)         # [N,1,1,1]
    x = x * contrast + mean * (1.0 - contrast)

    # Brightness shift (additive)
    if brightness != 0.0:
        x = x + brightness

    # Gaussian noise
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std

    # Clamp to valid range
    x = x.clamp(0.0, 1.0)

    return x

def gaussian_blur_tensor(img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Wrapper for torchvision's gaussian_blur"""
    return tv_gaussian_blur(img, kernel_size, sigma)

def minmax_invert(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = img.amin(dim=(-2, -1), keepdim=True)
    mx = img.amax(dim=(-2, -1), keepdim=True)
    return (img - mn) / (mx - mn + eps)

def augment_image(image, contrast_range=(-0.4, 0.4), brightness_range=(-0.4, 0.4)):
    """
    Augments a numpy image with rotation, perspective transform, resizing, contrast, and brightness.
    Returns: (augmented_shade, augmented_raw)
    """
    height, width = image.shape
    shape = 245
    x = np.random.randint(0, width - shape)
    y = np.random.randint(0, height - shape)
    angle = np.random.uniform(-30, 30)
    
    image = rotate(image, angle, reshape=False, mode='constant', cval=120)
    image = image[y:y+shape, x:x+shape]

    pts1 = np.float32([[0, 0], [shape, 0], [0, shape], [shape, shape]])
    perturb = np.random.randint(-60, 60, (4, 2)).astype(np.float32)
    pts2 = pts1 + perturb
    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, M, (shape, shape), borderMode=cv2.BORDER_CONSTANT, borderValue=120)
    
    scale_factor = random.uniform(0.8, 1.2)
    new_size = (int(shape * scale_factor), int(shape * scale_factor))
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    
    contrast_factor = random.uniform(1 + contrast_range[0], 1 + contrast_range[1])
    image_shades = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    
    brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
    image_shades = cv2.convertScaleAbs(image_shades, alpha=1, beta=brightness_factor * 255)
    
    return image_shades, image
