import torch
import torch.nn as nn
import torch.nn.functional as F

from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d as MGNCC
from diffdrr.metrics import GradientNormalizedCrossCorrelation2d as GNCC
from diffdrr.metrics import NormalizedCrossCorrelation2d as NCC

from src.utils.image_processing import rotation_6d_to_matrix
from src.core.layers import Sobel
from src.utils import config


class PositionLoss(nn.Module):
    def __init__(self, scales=(0.01, 0.001, 0.01)):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales).float())  # (3,)

    def forward(self, vector_pred, vector_gt):
        # Normalize translations
        if vector_gt.device != vector_pred.device:
            vector_gt = vector_gt.to(vector_pred.device)

        vec_pred_norm = vector_pred.clone()
        vec_gt_norm = vector_gt.clone()

        vec_pred_norm[:, 3:] = vector_pred[:, 3:] * self.scales
        vec_gt_norm[:, 3:] = vector_gt[:, 3:] * self.scales

        # Losses
        rot_loss = F.mse_loss(vec_pred_norm[:, :3], vec_gt_norm[:, :3])
        trans_loss = F.mse_loss(vec_pred_norm[:, 3:], vec_gt_norm[:, 3:])

        total_loss = rot_loss + trans_loss

        return total_loss

class MaskedSSIMLoss(nn.Module):
    """
    Masked Structural Similarity (SSIM) Loss.

    This loss computes the SSIM index between a predicted image and a target image,
    and applies a spatial mask to focus the loss only on selected regions (e.g., masked patches).

    The final loss is:
        L = mean_over_batch( sum((1 - SSIM_map) * mask) / sum(mask) )

    Notes:
    - Masking is applied AFTER SSIM computation (approximation, but standard in practice).
    - Inputs are expected to be normalized to [0, 1].

    Args:
        window_size (int): Size of the Gaussian window used for SSIM computation.
        sigma (float): Standard deviation of the Gaussian window.
        channels (int): Number of input channels.
        data_range (float): Value range of the input images (default: 1.0 for [0,1]).
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
        data_range: float = 1.0,
    ):
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels

        # SSIM stability constants (from original paper)
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2

        # Create Gaussian window and register as buffer (not trainable, moves with model)
        window = self._create_window(window_size, sigma, channels)
        self.register_buffer("window", window)

    def _create_window(self, window_size: int, sigma: float, channels: int) -> torch.Tensor:
        """
        Creates a 2D Gaussian window used for SSIM computation.

        Args:
            window_size (int): Size of the Gaussian kernel.
            sigma (float): Standard deviation.
            channels (int): Number of channels.

        Returns:
            torch.Tensor: Gaussian kernel of shape (channels, 1, window_size, window_size)
        """

        # Create 1D Gaussian kernel
        gauss_1d = torch.signal.windows.gaussian(window_size, std=sigma)
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Convert to 2D Gaussian kernel via outer product
        gauss_2d = torch.outer(gauss_1d, gauss_1d)
        gauss_2d = gauss_2d / gauss_2d.sum()

        # Expand to match conv2d grouped format: (C, 1, H, W)
        window = gauss_2d.expand(channels, 1, window_size, window_size)

        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the masked SSIM loss.

        Args:
            pred (torch.Tensor): Predicted image, shape (B, C, H, W)
            target (torch.Tensor): Ground truth image, shape (B, C, H, W)
            pixel_mask (torch.Tensor): Binary mask, shape (B, 1, H, W)
                                        1 = include in loss, 0 = ignore

        Returns:
            torch.Tensor: Scalar loss value
        """

        # Ensure mask is float for multiplication
        pixel_mask = pixel_mask.float()

        # Move window to correct device and dtype
        window = self.window.to(pred.device, dtype=pred.dtype)

        padding = self.window_size // 2
        channels = pred.size(1)

        # ---- Compute local means using Gaussian filter ----
        mu1 = F.conv2d(pred, window, padding=padding, groups=channels)
        mu2 = F.conv2d(target, window, padding=padding, groups=channels)

        # ---- Compute squared means and cross mean ----
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu12 = mu1 * mu2

        # ---- Compute local variances and covariance ----
        sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu12

        # ---- Compute SSIM map ----
        ssim_map = ((2 * mu12 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )

        # ---- Convert SSIM to loss and apply mask ----
        loss_map = (1 - ssim_map) * pixel_mask

        # ---- Normalize per sample (avoid bias due to mask size) ----
        loss_per_sample = loss_map.flatten(1).sum(dim=1) / pixel_mask.flatten(1).sum(dim=1).clamp_min(1.0)

        # ---- Return batch mean ----
        return loss_per_sample.mean()

class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for same-patient DRR pairs.

    This loss encourages the encoder to produce similar feature representations
    for different DRRs originating from the same patient.

    Given:
        - Anchor image x_i
        - Positive image x_j (same patient, different pose)

    The encoder produces:
        f_i = encoder(x_i)
        f_j = encoder(x_j)

    The loss is defined as:
        L = mean(1 - cosine_similarity(f_i, f_j))

    Notes:
    - Operates on feature vectors (not images)
    - Requires paired inputs (anchor, positive)
    - Uses global average pooling to obtain compact representations
    - Does NOT include negative samples (this is NOT contrastive learning)

    Args:
        eps (float): Small value for numerical stability in cosine similarity
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        feat_anchor: torch.Tensor,
        feat_positive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes cosine similarity loss.

        Args:
            feat_anchor (torch.Tensor): Feature map from anchor images (B, C, H, W) or (B, C)
            feat_positive (torch.Tensor): Feature map from positive images (B, C, H, W) or (B, C)

        Returns:
            torch.Tensor: Scalar loss value
        """

        # ---- Ensure features are vectors ----
        if feat_anchor.dim() == 4:
            # Global average pooling → (B, C, 1, 1)
            feat_anchor = F.adaptive_avg_pool2d(feat_anchor, 1).flatten(1)
            feat_positive = F.adaptive_avg_pool2d(feat_positive, 1).flatten(1)

        # ---- Normalize features ----
        feat_anchor = F.normalize(feat_anchor, dim=1, eps=self.eps)
        feat_positive = F.normalize(feat_positive, dim=1, eps=self.eps)

        # ---- Compute cosine similarity ----
        cos_sim = F.cosine_similarity(feat_anchor, feat_positive, dim=1)

        # ---- Convert similarity to loss ----
        loss = 1.0 - cos_sim

        return loss.mean()

class mGNCCLoss(nn.Module):
    """
    Multi-Scale Gradient Normalized Cross Correlation (mGNCC) Loss.
    This loss computes the gain between projection and target C-arm gradients
    across multiple scales using a set of differentiable Sobel filters.
    """
    def __init__(self, scales=(128, 64, 32, 16, 8), weights=None):
        super().__init__()
        self.scales = scales
        self.sobel = Sobel()

        self.n_levels = len(self.scales)

        if weights is None:
            weights = torch.ones(self.n_levels, device=config.DEVICE)

        # self.gain_fn = MGNCC(patch_sizes=self.scales, patch_weights=weights)
        # self.gain_fn = GNCC()
        # self.gain_fn = GNCC(patch_size=32)
        # self.gain_fn = NCC(patch_size=8)
        self.gain_fn = NCC()

    def forward(self, projection: torch.Tensor, carm: torch.Tensor, kernel=1) -> torch.Tensor:
        """
        Computes the MNCC gain.
        
        Args:
            projection (torch.Tensor): Predicted rendered projection.
            carm (torch.Tensor): Ground truth (target) C-arm projection.
            kernel: Optional kernel masking or weighting.
            weights: Optional per-scale weighting for MNCC.
            
        Returns:
            torch.Tensor: The average gain over all scales.
        """
        # projection_gradients = self.sobel(projection) * kernel
        # carm_gradients = self.sobel(carm) * kernel
        from torchvision.transforms.functional import gaussian_blur
        projection = gaussian_blur(projection, 5, 1.0)
        carm = gaussian_blur(carm, 5, 1.0)
        porj_mag, proj_ori = self.sobel(projection, return_orientation=True) * kernel
        carm_mag, carm_ori = self.sobel(carm, return_orientation=True) * kernel

        # mncc_gain = (self.gain_fn(porj_mag, carm_mag) + self.gain_fn(proj_ori, carm_ori)) / 2
        # mncc_gain = self.gain_fn(porj_mag, carm_mag)
        mncc_gain = self.gain_fn(projection, carm)
        # return mncc_gain / self.n_levels
        return mncc_gain

class LocalNCC(nn.Module):
    def __init__(self, patch_size=9, stride=1, eps=1e-5, min_valid_ratio=0.3):
        """
        Local NCC using unfold (reference implementation).

        Args:
            patch_size (int)
            stride (int)
            eps (float)
            min_valid_ratio (float): minimum fraction of valid pixels in a patch
        """
        super().__init__()
        self.k = patch_size
        self.stride = stride
        self.eps = eps
        self.min_valid = min_valid_ratio * (patch_size ** 2)

    def forward(self, x, y, mask=None):
        """
        Args:
            x, y: (B, C, H, W)
            mask: (B, 1, H, W) or (B, C, H, W), optional

        Returns:
            (B,) LNCC per image
        """
        assert x.shape == y.shape, "x and y must have same shape"
        B, C, H, W = x.shape
        k = self.k

        # ---- Unfold patches ----
        x_p = F.unfold(x, kernel_size=k, stride=self.stride)  # (B, C*k*k, L)
        y_p = F.unfold(y, kernel_size=k, stride=self.stride)

        L = x_p.shape[-1]

        # reshape → (B, C, k*k, L)
        x_p = x_p.view(B, C, k*k, L)
        y_p = y_p.view(B, C, k*k, L)

        # ---- Handle mask ----
        if mask is not None:
            mask = mask.to(x.device).to(x.dtype)

            if mask.shape[1] == 1:
                mask = mask.expand(-1, C, -1, -1)

            m_p = F.unfold(mask, kernel_size=k, stride=self.stride)
            m_p = m_p.view(B, C, k*k, L)

            # valid pixel count per patch
            count = m_p.sum(dim=[2, 3], keepdim=True).clamp(min=1.0)

            # ---- Masked means ----
            x_mu = (x_p * m_p).sum(dim=[2, 3], keepdim=True) / count
            y_mu = (y_p * m_p).sum(dim=[2, 3], keepdim=True) / count
            x_mu2 = x_p.mean(dim=[-1, -2], keepdim=True)
            # print(f"x_mu: {x_mu}")
            # print(f"x_mu2: {x_mu2}")
            # print(f"x mean: {x_mu*count/(k*k*L)}")
            # print(f"count: {count}")

            # ---- Centered ----
            x_c = (x_p - x_mu) * m_p
            y_c = (y_p - y_mu) * m_p
            # print(f"x_c: {x_c[0, 0, 50:55, :]}")
            # print(x_c.mean(dim=[2,3]))
            # print(f"x_c: {x_c[0,0,:,533]}")
            # print(x_c.sum(dim=[2, 3], keepdim=True))

            # ---- Variance / covariance ----
            x_var = (x_c ** 2).sum(dim=[2, 3]) / count
            y_var = (y_c ** 2).sum(dim=[2, 3]) / count
            
            x_var2 = x_c.var(dim=[-1, -2], keepdim=True, correction=0) + self.eps
            # print(f"x_var: {x_var}")
            # print(f"x_var2: {x_var2}")
            # print(f"x_var: {x_var*count/(k*k*L)}")
            # cov_xy = (x_c * y_c).sum(dim=2)

            # # ---- NCC ----
            # ncc = cov_xy / (torch.sqrt(var_x * var_y + self.eps))

            # # ---- Reject weak patches ----
            # valid = (count.squeeze(2) > self.min_valid)
            # ncc = ncc * valid

            # valid_counts = valid.sum(dim=(1, 2)).clamp(min=1)
            # ncc = ncc.sum(dim=(1, 2)) / valid_counts


            # # ---- effective pixel ratio per patch ----
            # valid = m_p.sum(dim=2, keepdim=True).clamp(min=1.0)
            # norm_factor = (k * k) / valid  # correction for mean/var bias

            # # ---- apply mask ----
            # x_p = x_p * m_p
            # y_p = y_p * m_p

            # # ---- mean (corrected to preserve .mean semantics) ----
            # x_mu = x_p.mean(dim=2, keepdim=True) * norm_factor
            # y_mu = y_p.mean(dim=2, keepdim=True) * norm_factor

            # # ---- variance (corrected similarly) ----
            # x_var = x_p.var(dim=2, keepdim=True, correction=0) * norm_factor
            # y_var = y_p.var(dim=2, keepdim=True, correction=0) * norm_factor

            # ---- normalize ----
            x_std = torch.sqrt(x_var + self.eps)
            y_std = torch.sqrt(y_var + self.eps)

            x_n = x_c / x_std
            y_n = y_c / y_std

            # ---- correlation (same structure as yours) ----
            score = torch.einsum("b...,b...->b", x_n, y_n)

            # ---- normalize ----
            norm = m_p.sum(dim=[1, 2, 3]).clamp(min=1.0)
            norm = m_p.sum()

            ncc = score / norm
        else:
            # ---- Standard LNCC (no mask) ----
            # mean_x = x_p.mean(dim=2, keepdim=True)
            # mean_y = y_p.mean(dim=2, keepdim=True)

            # x_c = x_p - mean_x
            # y_c = y_p - mean_y

            # var_x = (x_c ** 2).sum(dim=2)
            # var_y = (y_c ** 2).sum(dim=2)
            # cov_xy = (x_c * y_c).sum(dim=2)

            # ncc = cov_xy / (torch.sqrt(var_x * var_y + self.eps))
            # ncc = ncc.mean(dim=(1, 2))

            # Normalize x
            x_mu = x_p.mean(dim=[-1, -2], keepdim=True)
            x_var = x_p.var(dim=[-1, -2], keepdim=True, correction=0) + self.eps
            x_std = x_var.sqrt()
            x_normalized = (x_p - x_mu) / x_std

            # Normalize y
            y_mu = y_p.mean(dim=[-1, -2], keepdim=True)
            y_var = y_p.var(dim=[-1, -2], keepdim=True, correction=0) + self.eps
            y_std = y_var.sqrt()
            y_normalized = (y_p - y_mu) / y_std

            score = torch.einsum("b...,b...->b", x_normalized, y_normalized)
            score /= C * H * W * k * k
            ncc = score

        return ncc

def compute_geodesic_distance(R1, R2):
    """
    Calculates Geodesic difference in Radians mapped accurately upon SO(3) domain
    R1, R2: Both sizes (B, 3, 3)
    """
    if len(R1.shape) == 3:
        R = torch.bmm(R1, R2.transpose(1, 2))
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    else:
        R = R1.transpose(-1, -2) @ R2
        trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)

    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_theta)

def poseConsistencyLoss(pred, repeats: int = 4, weight : float = 0.005, return_sum: bool = True):
    """
    pred: (B * V, 9) → [6D rot | 3D trans]
    """
    Bv, D = pred.shape
    V = repeats
    if V <= 1:
        return torch.tensor(0.0, device=pred.device, requires_grad=False)
    assert Bv % V == 0
    B = Bv // V

    pred = pred.view(B, V, D)

    rot6d = pred[..., :6]   # (B, V, 6)
    trans = pred[..., 6:]   # (B, V, 3)

    # convert rotations to matrices
    R = rotation_6d_to_matrix(rot6d)  # (B, V, 3, 3)

    # pairwise comparisons
    R1 = R[:, :, None]   # (B, V, 1, 3, 3)
    R2 = R[:, None, :]   # (B, 1, V, 3, 3)

    rot_dist = compute_geodesic_distance(R1, R2)  # (B, V, V)

    t1 = trans[:, :, None, :]  # (B, V, 1, 3)
    t2 = trans[:, None, :, :]  # (B, 1, V, 3)

    trans_dist = torch.norm(t1 - t2, dim=-1)  # (B, V, V)

    # remove diagonal (same-view comparisons)
    mask = ~torch.eye(V, dtype=torch.bool, device=pred.device)
    rot_loss = rot_dist[:, mask].mean()
    trans_loss = trans_dist[:, mask].mean()

    if return_sum:
        return rot_loss + weight * trans_loss
    else:
        return rot_loss, weight * trans_loss
