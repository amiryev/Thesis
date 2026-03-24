import os
import sys
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

from old_code.utils import Sobel, DirectionalMambaBlock

# NOTE: We keep imports light here; DRR/CT loading now happens in the dataset (train.py).

class XrayEncoder(nn.Module):
    """
    Masked reconstruction autoencoder:
      input: x (B,1,H,W)   # raw projection in [0,1]
      masking: patch-wise mask on x (zeros on masked patches)
      backbone: concat [x_masked, Sobel mag, Sobel orient] -> VGG16.features -> Mamba
      decoder: upsample to (B,1,H,W)
      output: recon (B,1,H,W), pixel_mask (B,1,H,W), features
    """
    def __init__(
        self,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        size: int = 128,
        patch_size: int = 32,
    ):
        super().__init__()
        self.device = device
        self.size = size
        self.ps = patch_size

        assert size % patch_size == 0, "size must be divisible by patch_size"
        self.num_patches_h = size // patch_size
        self.num_patches_w = size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Inputs: (x_masked, sobel_mag, sobel_orient) -> 3 channels to match VGG
        self.sobel = Sobel()

        # Encoder backbone
        self.encoder = vgg16(weights=VGG16_Weights.DEFAULT).features  # (B, 512, 4, 4) for 128x128
        self.encoded_shape = (512, 4, 4)  # (C, H, W)

        # Mamba over flattened spatial tokens with learned positional encoding
        C, H, W = self.encoded_shape
        self.positional_encoding = nn.Parameter(
            torch.zeros(H * W, C)
        )
        torch.nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        self.mask_token = nn.Parameter(torch.zeros(C))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        directions = [
            "tl_row",
            "br_col", 
            "tr_row",  
            "bl_col",
            "br_row",  
            "tl_col",  
            "bl_row",
            "tr_col",   
        ]

        self.mamba = nn.ModuleList(
            [DirectionalMambaBlock(d_model=C, H=H, W=W, mode=m) for m in directions]
        )

        # Lightweight decoder to 1xHxW
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),  # 32->64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  16,  kernel_size=4, stride=2, padding=1),  # 64->128
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # target images normalized to [0,1]
        )

    def _make_patch_mask(self, B: int, mask_ratio: float):
        """Return patch-level mask as expanded pixel mask (B,1,H,W)."""
        ps = self.ps
        Hp, Wp = self.num_patches_h, self.num_patches_w
        L = Hp * Wp
        num_mask = int(mask_ratio * L)

        mask_patch = torch.zeros((B, L), device=self.device, dtype=torch.float32)
        for b in range(B):
            idx = torch.randperm(L, device=self.device)[:num_mask]
            mask_patch[b, idx] = 1.0

        mask_grid = mask_patch.view(B, 1, Hp, Wp)
        pixel_mask = mask_grid.repeat_interleave(ps, dim=2).repeat_interleave(ps, dim=3)
        return pixel_mask, mask_patch  # (B,1,H,W), (B,L)

    def forward(self, x: torch.Tensor, mask_ratio: float):
        """
        x: (B,1,H,W) in [0,1]
        returns:
          recon: (B,1,H,W)
          pixel_mask: (B,1,H,W) with 1 where masked
          features: encoder features before decoder (B,512,4,4)
        """
        B, C, H, W = x.shape
        assert C == 1, "Expect (B,1,H,W) grayscale input"

        pixel_mask, patch_mask = self._make_patch_mask(B, mask_ratio)

        feats = self.encode(x, patch_mask=patch_mask)

        # Decode
        recon = self.decode(feats)                               # (B,1,H,W) ~[0,1]
        return recon, pixel_mask, feats


    def encode(self, x: torch.Tensor, kernel=None, patch_mask: torch.Tensor = None):
        mag, orient = self.sobel(x, return_orientation=True)  # (B,1,H,W) each
        if kernel is not None:
            x3 = torch.cat([x, mag, orient], dim=1) * kernel
        else:              # (B,3,H,W)
            x3 = torch.cat([x, mag, orient], dim=1)
        x = self.encoder(x3)  # (B,512,4,4)

        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)  # (B,L,C)

        if patch_mask is not None:
            mask = patch_mask.bool()
            tokens[mask] = self.mask_token
        
        tokens = tokens + self.positional_encoding.unsqueeze(0)  # (B,L,C)
        
        for block in self.mamba:
            tokens = block(tokens)

        tokens = tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return tokens
        
    def decode(self, x: torch.Tensor):
        return self.decoder(x)