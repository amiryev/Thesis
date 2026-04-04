import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur

import numpy as np
import random
import cv2
from scipy.ndimage import rotate

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


# class RandomGaussianNoise(torch.nn.Module):
#     def __init__(self, std_range=(0.0, 0.05)):
#         super().__init__()
#         self.min_std, self.max_std = std_range

#     def forward(self, img):
#         std = random.uniform(self.min_std, self.max_std)
#         if std == 0:
#             return img
#         noise = torch.randn_like(img) * std
#         return torch.clamp(img + noise, 0.0, 1.0)

# class RandomGamma(torch.nn.Module):
#     def __init__(self, range=(0.5, 2.0)):
#         super().__init__()
#         self.min_gamma, self.max_gamma = range

#     def forward(self, img):
#         gamma = random.uniform(self.min_gamma, self.max_gamma)
#         return TF.adjust_gamma(img, gamma)

# class RandomGaussianBlur(torch.nn.Module):
#     def __init__(self, sigma_range=(0.0, 1.5), kernel_size=9):
#         super().__init__()
#         self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma_range)

#     def forward(self, img):
#         return self.blur(img)

# class RandomContrast(torch.nn.Module):
#     def __init__(self, range=(0.7, 1.3)):
#         super().__init__()
#         self.min_c, self.max_c = range

#     def forward(self, img):
#         c = random.uniform(self.min_c, self.max_c)
#         return TF.adjust_contrast(img, c)

# --- Rotational Math Utilities ---

def euler_angles_to_matrix(euler_angles, convention="ZXY"):
    """
    Convert (B, 3) Euler angles in radians to a (B, 3, 3) rotation matrix.
    Convention "ZXY" logic applies sequential product: Rz * Rx * Ry
    """
    z, x, y = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)
    
    rx = torch.stack([
        torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
        torch.zeros_like(x), cx, -sx,
        torch.zeros_like(x), sx, cx
    ], dim=1).reshape(-1, 3, 3)
    
    ry = torch.stack([
        cy, torch.zeros_like(y), sy,
        torch.zeros_like(y), torch.ones_like(y), torch.zeros_like(y),
        -sy, torch.zeros_like(y), cy
    ], dim=1).reshape(-1, 3, 3)
    
    rz = torch.stack([
        cz, -sz, torch.zeros_like(z),
        sz, cz, torch.zeros_like(z),
        torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z)
    ], dim=1).reshape(-1, 3, 3)
    
    if convention == "ZXY":
        return torch.bmm(rz, torch.bmm(rx, ry))
    return torch.bmm(rx, torch.bmm(ry, rz)) # Fallback implementation

def matrix_to_rotation_6d(matrix):
    """
    Grab the first two columns spanning the continuous space.
    matrix: (B, 3, 3)
    Returns: (B, 6)
    """
    return matrix[:, :, :2].reshape(-1, 6)

def rotation_6d_to_matrix(d6):
    """
    Differentiable step to form a valid orthogonal 3x3 rotation matrix using Zhou's Continuous 6D formula.
    d6: (B, 6) Raw coordinates of the 2 orthogonal basis vectors representation 
    Returns: (B, 3, 3) Orientation matrix
    """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    
    x = F.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    # z = y_raw - (x * y_raw).sum(-1, keepdim=True) * x
    z = F.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    
    return torch.stack((x, y, z), dim=-1)

def matrix_to_euler_angles(matrix, convention="ZXY", eps=1e-6):
    """
    Convert rotation matrix (B, 3, 3) to Euler angles (B, 3)
    matching euler_angles_to_matrix with ZXY convention:
        R = Rz * Rx * Ry
    
    Returns angles in radians: (z, x, y)
    """
    if convention != "ZXY":
        raise NotImplementedError(f"Convention {convention} not supported")

    R = matrix
    r00, r01, r02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r10, r11, r12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r20, r21, r22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # x = asin(r21)
    x = torch.asin(torch.clamp(r21, -1.0 + eps, 1.0 - eps))
    cx = torch.cos(x)

    # Detect gimbal lock
    gimbal_lock = torch.abs(cx) < eps

    # Standard case
    z = torch.atan2(-r01, r11)
    y = torch.atan2(-r20, r22)

    # Gimbal lock fallback
    z_gl = torch.atan2(r10, r00)
    y_gl = torch.zeros_like(y)

    z = torch.where(gimbal_lock, z_gl, z)
    y = torch.where(gimbal_lock, y_gl, y)

    return torch.stack([z, x, y], dim=1)

class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, std_range=(0.0, 0.05)):
        super().__init__()
        self.min_std, self.max_std = std_range

    def forward(self, img):
        B = img.shape[0]
        std = torch.empty(B, 1, 1, 1, device=img.device).uniform_(self.min_std, self.max_std)
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0.0, 1.0)
    
class RandomGamma(torch.nn.Module):
    def __init__(self, range=(0.5, 2.0)):
        super().__init__()
        self.min_gamma, self.max_gamma = range

    def forward(self, img):
        B = img.shape[0]
        gamma = torch.empty(B, 1, 1, 1, device=img.device).uniform_(self.min_gamma, self.max_gamma)
        return torch.clamp(img ** gamma, 0.0, 1.0)
    
class RandomContrast(torch.nn.Module):
    def __init__(self, range=(0.7, 1.3)):
        super().__init__()
        self.min_c, self.max_c = range

    def forward(self, img):
        B = img.shape[0]
        c = torch.empty(B, 1, 1, 1, device=img.device).uniform_(self.min_c, self.max_c)
        mean = img.mean(dim=(2, 3), keepdim=True)
        return torch.clamp((img - mean) * c + mean, 0.0, 1.0)
    
class RandomGaussianBlur(torch.nn.Module):
    def __init__(self, sigma_range=(0.0, 1.5), kernel_size=9):
        super().__init__()
        self.min_sigma, self.max_sigma = sigma_range
        self.kernel_size = kernel_size
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma_range)

    def forward(self, imgs):
        """
        img: (B, C, H, W)
        """
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        imgs_out = torch.stack([self.blur(img) for img in imgs])
        return imgs_out
    
class RandomSpatialJitter(torch.nn.Module):
    """
    Small random affine transform that preserves pose label validity.
    Translation and scale are small enough that the anatomical region 
    stays in frame and the projection geometry is only minimally affected.
    """
    def __init__(self, max_translate_frac=0.03, scale_range=(0.95, 1.05)):
        super().__init__()
        self.max_t = max_translate_frac
        self.scale_range = scale_range

    def forward(self, imgs):  # (B, C, H, W)
        B, C, H, W = imgs.shape
        out = []
        for i in range(B):
            tx = torch.FloatTensor(1).uniform_(-self.max_t, self.max_t).item() * W
            ty = torch.FloatTensor(1).uniform_(-self.max_t, self.max_t).item() * H
            sc = torch.FloatTensor(1).uniform_(*self.scale_range).item()
            out.append(TF.affine(imgs[i], angle=0, translate=[tx, ty], scale=sc, shear=0))
        return torch.stack(out)