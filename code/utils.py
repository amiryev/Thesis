import cv2
import numpy as np
import random
import math

from scipy.ndimage import rotate
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
import random
from scipy.ndimage import rotate
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

class Sobel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=2,  
            kernel_size=3,
            stride=1,
            padding=1,  
            bias=False,
        )

        Gx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(torch.float32)
        Gy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(torch.float32)
        G = torch.stack([Gx, Gy]).unsqueeze(1)
        self.filter.weight = torch.nn.Parameter(G, requires_grad=False)

    def forward(self, img, return_orientation=False):
        x = self.filter(img)
        gx, gy = torch.chunk(x, 2, dim=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2)
        if return_orientation:
            orientation = torch.atan2(gy, gx) / math.pi
            return magnitude, orientation
        return magnitude


class MambaBlock(nn.Module):
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2, use_residual=True):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.use_residual = use_residual
    

    def forward(self, x):
        residual = x if self.use_residual else 0
        x = self.mamba(x)               
        x = self.norm(x)         
        x = self.linear(x)       
        x = self.activation(x)        
        return x + residual          


class DirectionalMambaBlock(nn.Module):
    """
    Applies a MambaBlock along a fixed scan order of HxW tokens.
    Input/Output: tokens (B, L, C) in row-major when entering/leaving.
    """
    def __init__(self, d_model: int, H: int, W: int, mode: str, d_state: int = None):
        super().__init__()
        if d_state is None:
            d_state = H * W
        self.block = MambaBlock(d_model=d_model, d_state=d_state)

        L = H * W
        order = []

        def idx(r, c): return r * W + c

        if mode == "tl_row":   # top-left, row snake
            for r in range(H):
                cols = range(W) if r % 2 == 0 else range(W - 1, -1, -1)
                for c in cols:
                    order.append(idx(r, c))

        elif mode == "tr_row": # top-right, row snake
            for r in range(H):
                cols = range(W - 1, -1, -1) if r % 2 == 0 else range(W)
                for c in cols:
                    order.append(idx(r, c))

        elif mode == "bl_row": # bottom-left, row snake upwards
            for r_off in range(H):
                r = H - 1 - r_off
                cols = range(W) if r_off % 2 == 0 else range(W - 1, -1, -1)
                for c in cols:
                    order.append(idx(r, c))

        elif mode == "br_row": # bottom-right, row snake upwards
            for r_off in range(H):
                r = H - 1 - r_off
                cols = range(W - 1, -1, -1) if r_off % 2 == 0 else range(W)
                for c in cols:
                    order.append(idx(r, c))

        # ---- Column snakes ----
        elif mode == "tl_col": # top-left, column snake
            for c in range(W):
                rows = range(H) if c % 2 == 0 else range(H - 1, -1, -1)
                for r in rows:
                    order.append(idx(r, c))

        elif mode == "tr_col": # top-right, column snake
            for c_off in range(W):
                c = W - 1 - c_off
                rows = range(H) if c_off % 2 == 0 else range(H - 1, -1, -1)
                for r in rows:
                    order.append(idx(r, c))

        elif mode == "bl_col": # bottom-left, column snake upwards
            for c in range(W):
                rows = range(H - 1, -1, -1) if c % 2 == 0 else range(H)
                for r in rows:
                    order.append(idx(r, c))

        elif mode == "br_col": # bottom-right, column snake upwards
            for c_off in range(W):
                c = W - 1 - c_off
                rows = range(H - 1, -1, -1) if c_off % 2 == 0 else range(H)
                for r in rows:
                    order.append(idx(r, c))

        else:
            raise ValueError(f"Unknown mode: {mode}")


        perm = torch.tensor(order, dtype=torch.long)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(L, dtype=torch.long)

        self.register_buffer("perm", perm, persistent=False)
        self.register_buffer("inv_perm", inv, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L, C) row-major
        x = tokens.index_select(dim=1, index=self.perm)     # directional order
        x = self.block(x)                                   # Mamba along that order
        x = x.index_select(dim=1, index=self.inv_perm)      # back to row-major
        return x


class PositionLoss(nn.Module):
    def __init__(self, scales=(0.01, 0.001, 0.01)):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales).float())  # (3,)

    def forward(self, vector_pred, vector_gt):
        # Verify GT and prediction are on the same device
        if vector_gt.device != vector_pred.device:
            vector_gt = vector_gt.to(vector_pred.device)

        # Normalize translations
        vec_pred_norm = vector_pred.clone()
        vec_gt_norm = vector_gt.clone()

        vec_pred_norm[:, 3:] = vector_pred[:, 3:] * self.scales
        vec_gt_norm[:, 3:] = vector_gt[:, 3:] * self.scales

        # Losses
        rot_loss = F.mse_loss(vec_pred_norm[:, :3], vec_gt_norm[:, :3])
        trans_loss = F.mse_loss(vec_pred_norm[:, 3:], vec_gt_norm[:, 3:])

        total_loss = rot_loss + trans_loss

        return total_loss

def minmax_invert(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = img.amin(dim=(-2, -1), keepdim=True)
    mx = img.amax(dim=(-2, -1), keepdim=True)
    return (img - mn) / (mx - mn + eps)




def augment_image(image, contrast_range=(-0.4, 0.4), brightness_range=(-0.4, 0.4)):
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
    
    return image_shades, image,


def find_registration(image1, image1_original, image2, pts1, pts2, keypoints1, keypoints2, matches):
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, maxIters=30000, ransacReprojThreshold=5)

    pts2_homogeneous = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    pts2_transformed = (H @ pts2_homogeneous.T).T
    pts2_transformed /= pts2_transformed[:, 2][:, np.newaxis]  # Normalize

    img1_warped = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))
    img1_warped_original = cv2.warpPerspective(image1_original, H, (image2.shape[1], image2.shape[0]))

    matched_img = cv2.drawMatches(
        image1, keypoints2, image2, keypoints1, matches, None,
        matchColor=(255, 0, 0),  # Red color for all matches
        singlePointColor=None,
        matchesMask=None,  # Draw all matches
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matched_img = cv2.drawMatches(
        image1, keypoints2, image2, keypoints1, matches, matched_img,
        matchColor=(0, 255, 0),  # Green color for inliers
        singlePointColor=None,
        matchesMask=mask.ravel().tolist(),  # Mask to draw only inliers
        flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Blend img2 onto img1
    mask_warped = (img1_warped > 0).astype(np.uint8)  # Mask of img2's valid region
    blended_img = image2 * (1 - mask_warped) + img1_warped * mask_warped  # Overlay img2 onto img1

    valid_mask = img1_warped_original > 120
    error = np.abs(image2 - img1_warped_original)
    filtered_squared_error = error[valid_mask] 

    if filtered_squared_error.size > 0:
        mse_loss = np.sum(filtered_squared_error)
    else:
        mse_loss = 0 

    return H, mask, img1_warped, matched_img, mask_warped, blended_img, int(mse_loss), error

def cosine_similarity_matches(descriptors2, descriptors1):
    similarity_matrix = cosine_similarity(descriptors2, descriptors1)
    matches = []
    for queryIdx in range(similarity_matrix.shape[0]):
        max_similarity_query = np.max(similarity_matrix[queryIdx, :])
        train_indices = np.where(similarity_matrix[queryIdx, :] == max_similarity_query)[0]
        for trainIdx in train_indices:
            # Step 2: For each train descriptor, find all query descriptors with the maximum similarity
            max_similarity_train = np.max(similarity_matrix[:, trainIdx])
            query_indices = np.where(similarity_matrix[:, trainIdx] == max_similarity_train)[0]
            # Step 3: Check for mutual best matches
            if queryIdx in query_indices:
                # Create a DMatch object
                distance = 1 - similarity_matrix[queryIdx, trainIdx]
                match = cv2.DMatch(_queryIdx=queryIdx, _trainIdx=trainIdx, _distance=distance)
                matches.append(match)
    return matches

def torch_to_cv2_keypoints(tensor):
    keypoints = []
    for i in range(tensor.shape[0]):
        x, y = tensor[i, 0].item(), tensor[i, 1].item()
        keypoints.append(cv2.KeyPoint(x, y, 1))
    return keypoints

def center_crop_radial_image(image):
    height, width = image.shape[:2]

    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 2

    max_square_half = int(radius / (2 ** 0.5)) 
    
    x1, y1 = center_x - max_square_half, center_y - max_square_half
    x2, y2 = center_x + max_square_half, center_y + max_square_half
    
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


def find_registration_modified(image0, image1, pts0, pts1, keypoints0, keypoints1, matches):
    H, mask = cv2.findHomography(pts1, pts0, cv2.USAC_MAGSAC, 5.0, confidence=0.999, maxIters=30000)

    img1_warped = cv2.warpPerspective(image1, H, (image0.shape[1], image0.shape[0]))

    matched_img = cv2.drawMatches(
        image0, keypoints0, image1, keypoints1, matches, None,
        matchColor=(255, 0, 0),  # Red color for all matches
        singlePointColor=None,
        matchesMask=None,  # Draw all matches
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matched_img = cv2.drawMatches(
        image0, keypoints0, image1, keypoints1, matches, matched_img,
        matchColor=(0, 255, 0),  # Green color for inliers
        singlePointColor=None,
        matchesMask=mask.ravel().tolist(),  # Mask to draw only inliers
        flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    mask_warped = (img1_warped > 0).astype(np.uint8)  # Mask of img2's valid region
    blended_img = image0 * (1 - mask_warped) + img1_warped * mask_warped  # Overlay img2 onto img1


    return H, mask, img1_warped, matched_img, mask_warped, blended_img

@torch.no_grad()
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