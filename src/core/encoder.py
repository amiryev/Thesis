from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

from src.core.layers import Sobel, DirectionalMambaBlock

class XrayEncoder(nn.Module):
    """
    Masked reconstruction autoencoder:
      input: x (B,1,H,W)   # raw projection in [0,1]
      masking: patch-wise mask on x (zeros on masked patches)
      backbone: concat [x_masked, Sobel mag, Sobel orient] -> ResNet.features -> Mamba
      decoder: upsample to (B,1,H,W)
      output: recon (B,1,H,W), pixel_mask (B,1,H,W), features
    """
    def __init__(
        self,
        device=None,
        size: int = 128,
        patch_size: int = 32,
        output_channels: int = 8192,
        feature_mask:bool = False
    ):
        super().__init__()
        if device is None:
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.size = size
        self.patch_size = patch_size
        self.feature_mask = feature_mask
        # self.feature_mask = True
        
        if self.feature_mask:
            assert size % 32 == 0, "size must be divisible by patch_size"
            self.num_patches_h = size // 32
            self.num_patches_w = size // 32
            self.num_patches = self.num_patches_h * self.num_patches_w
        else:
            assert size % patch_size == 0, "size must be divisible by patch_size"
            self.num_patches_h = size // patch_size
            self.num_patches_w = size // patch_size
            self.num_patches = self.num_patches_h * self.num_patches_w

        # Inputs: (x_masked, sobel_mag, sobel_orient) -> 3 channels to match VGG
        self.sobel = Sobel()

        # Encoder backbone
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.size, self.size)
            feat = self.encoder(dummy)
            C, H, W = feat.shape[1:]  # skip batch dim  

        reduced_channels = output_channels // (H * W)
        self.channel_reduce = nn.Conv2d(C, reduced_channels, kernel_size=1)
        C = reduced_channels

        # Mamba over flattened spatial tokens with learned positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(H * W, C))
        torch.nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        # Learned mask token
        if self.feature_mask:
            self.mask_token = nn.Parameter(torch.zeros(C))
        else:
            self.mask_token = nn.Parameter(torch.zeros(self.size, self.size))

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
            nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1),  # 4->8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C//2, C//4, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C//4, C//8,  kernel_size=4, stride=2, padding=1),  # 16->32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C//8,  C//16,  kernel_size=4, stride=2, padding=1),  # 32->64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(C//16,  C//32,  kernel_size=4, stride=2, padding=1),  # 64->128
            nn.ReLU(inplace=True),
            nn.Conv2d(C//32, 1, kernel_size=3, padding=1),
            # nn.Conv2d(C//16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # target images normalized to [0,1]
        )

    def _make_feature_mask(self, B: int, mask_ratio: float):
        """Return patch-level mask as expanded pixel mask (B,1,H,W)."""
        Hp, Wp = self.num_patches_h, self.num_patches_w
        L = self.num_patches
        num_mask = int(mask_ratio * L)

        feature_mask = torch.zeros((B, L), device=self.device, dtype=torch.float32)
        for b in range(B):
            idx = torch.randperm(L, device=self.device)[:num_mask]
            feature_mask[b, idx] = 1.0

        mask_grid = feature_mask.view(B, 1, Hp, Wp)
        patch_mask = mask_grid.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)
        return patch_mask, feature_mask  # (B,1,H,W), (B,L)

    def forward(self, x: torch.Tensor, mask_ratio: float):
        """
        x: (B,1,H,W) in [0,1]
        returns:
          recon: (B,1,H,W)
          pixel_mask: (B,1,H,W) with 1 where masked
          features: encoder features before decoder (B,C,Hp,Wp)
        """
        B, C, H, W = x.shape
        assert C == 1, "Expect (B,1,H,W) grayscale input"

        patch_mask, feature_mask = self._make_feature_mask(B, mask_ratio)

        if self.feature_mask:
            mask = feature_mask
        else:
            mask = patch_mask

        feats = self.encode(x, patch_mask=mask)

        # Decode
        recon = self.decode(feats)                               # (B,1,H,W) ~[0,1]
        return recon, mask, feats


    def encode(self, x: torch.Tensor, kernel=None, patch_mask: torch.Tensor = None):
        
        # mask input image
        if (patch_mask is not None) and (self.feature_mask is False):
            x = x * (1 - patch_mask) + (self.mask_token * patch_mask)
            # x = x * (1 - patch_mask)

        # Sobel from masked image (important!)
        mag, orient = self.sobel(x, return_orientation=True)
        # mag, orient = x.clone(), x.clone()
        
        if kernel is not None:
            # Ensure kernel matches x shape if needed, mostly used for CRM masking
            x3 = torch.cat([x, mag, orient], dim=1) * kernel
        else:
            x3 = torch.cat([x, mag, orient], dim=1) # (B,3,H,W)
        
        x = self.encoder(x3) # (B,256,H/32,W/32)

        x = self.channel_reduce(x)

        B, C, H, W = x.shape
        tokens = x.view(B, C, H * W).permute(0, 2, 1)  # (B,L,C)

        if (patch_mask is not None) and (self.feature_mask is True):
            mask = patch_mask.bool()
            tokens[mask] = self.mask_token
            # mask = patch_mask.unsqueeze(-1).float()   # (B, L, 1)
            # tokens = tokens * (1 - mask) + self.mask_token * mask

        tokens = tokens + self.positional_encoding.unsqueeze(0)  # (B,L,C)
        
        for block in self.mamba:
            tokens = block(tokens)

        tokens = tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return tokens
        
    def decode(self, x: torch.Tensor):
        return self.decoder(x)
