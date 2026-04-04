import torch
import torch.nn as nn
import torchvision.models as models

from src.core.layers import SpatialAttentionPool, DirectionalMambaBlock, AttentionPool
from src.utils import config

class SobelConv(nn.Module):
    """
    Computes horizontal and vertical gradients of the input DRR and returns a 
    3-channel tensor combining the raw input, gradient magnitude, and orientation.
    Filters are fully learnable initialized to Sobel weights.
    """
    def __init__(self):
        super().__init__()
        # Learnable 3x3 filters for X and Y gradients
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # Initialize with literal Sobel weights
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        
        with torch.no_grad():
            self.conv_x.weight.copy_(sobel_x)
            self.conv_y.weight.copy_(sobel_y)

    def forward(self, x, four_c=False):
        """
        Input: 
            x (torch.Tensor): Tensor of shape (B, 1, H, W).
        Output:
            (torch.Tensor): Tensor of shape (B, 3, H, W) [raw, magnitude, orientation].
        """
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        
        eps = 1e-6
        magnitude = torch.sqrt(gx**2 + gy**2 + eps)
        if four_c == False:
            orientation = torch.atan2(gy, gx)
            return torch.cat([x, magnitude, orientation], dim=1)
        else:
            norm = magnitude.clamp(min=eps)
            cos_o = gx / norm
            sin_o = gy / norm
            # Normalize each channel to ~unit range
            mag_norm = magnitude / (magnitude.amax(dim=(-2,-1), keepdim=True) + eps)
            return torch.cat([x, mag_norm, cos_o, sin_o], dim=1) 

choose = 3
if choose == 1:
    class PoseRegressor(nn.Module):
        """
        Pose regressor mapping a single channel DRR image to a 6D continuous rotation
        and a 3D translation vector. Applies custom Sobel preprocess, ResNet18 backbone
        (re-initialized with GroupNorm), and an MLP prediction head.
        """
        def __init__(self, dropout=0.3):
            super().__init__()
            self.sobel = SobelConv()
            
            # Base ResNet18 Native
            # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            resnet = models.resnet18(weights=None)
            
            # Replace BatchNorm with GroupNorm for smaller batch sizes stability
            self.replace_bn_with_gn(resnet)
            
            # Standard input for ResNet18 is 3 channels, so we just enforce initialization.
            # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            # nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # Extract features up to GAP output
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            
            with torch.no_grad():
                dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                feat = self.backbone(dummy)
                C, H, W = feat.shape[1:]  # skip batch dim 

            # ResNet18 provides 512 channels out of GAP
            self.mlp = nn.Sequential(
                nn.Linear(C, C//2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//2, C//4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//4, 9)  # 6D rotation + 3D translation
            )


        def replace_bn_with_gn(self, module, num_groups=32):
            """
            Recursively replaces all nn.BatchNorm2d layers in a module with nn.GroupNorm.
            """
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    groups = min(num_groups, child.num_features)
                    gn = nn.GroupNorm(groups, child.num_features)
                    setattr(module, name, gn)
                else:
                    self.replace_bn_with_gn(child, num_groups)

        def forward(self, x):
            """
            Forward logic.
            
            Input:
                x (torch.Tensor): Input image batch of shape (B, 1, H, W)
                
            Output:
                rotation_6d (torch.Tensor): Continuous rotation map (B, 6)
                translation (torch.Tensor): Translation vector (B, 3)
            """
            # Feature processing
            x = self.sobel(x)            # (B, 3, H, W)
            x = self.backbone(x)         # (B, 512, 1, 1)
            x = x.view(x.size(0), -1)    # (B, 512)

            # Dense layers
            out = self.mlp(x)            # (B, 9)
            
            # Separation
            rotation_6d = out[:, :6]
            translation = out[:, 6:]
            
            return rotation_6d, translation
elif choose == 2:
    class PoseRegressor(nn.Module):
        """
        Pose regressor mapping a single channel DRR image to a 6D continuous rotation
        and a 3D translation vector. Applies custom Sobel preprocess, ResNet18 backbone
        (re-initialized with GroupNorm), and an MLP prediction head.
        """
        def __init__(self, dropout=0.3):
            super().__init__()
            self.sobel = SobelConv()
            
            # Base ResNet18 Native
            # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            resnet = models.resnet18(weights=None)
            
            # Replace BatchNorm with GroupNorm for smaller batch sizes stability
            self.replace_bn_with_gn(resnet)
            
            # Standard input for ResNet18 is 3 channels, so we just enforce initialization.
            # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            # nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # Extract features up to GAP output
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            
            with torch.no_grad():
                dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                feat = self.backbone(dummy)
                C, H, W = feat.shape[1:]  # skip batch dim 

            Q = 4
            CQ = C*Q
            self.spatial_attn = SpatialAttentionPool(C, num_queries=Q)

            # ResNet18 provides 512 channels out of GAP
            self.mlp = nn.Sequential(
                nn.Linear(CQ, CQ//4),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            self.rotation = nn.Sequential(
                nn.Linear(CQ//4, CQ//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(CQ//8, 6)  # 6D rotation + 3D translation
            )

            self.translation = nn.Sequential(
                nn.Linear(CQ//4, CQ//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(CQ//8, 3)  # 6D rotation + 3D translation
            )

        def replace_bn_with_gn(self, module, num_groups=32):
            """
            Recursively replaces all nn.BatchNorm2d layers in a module with nn.GroupNorm.
            """
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    groups = min(num_groups, child.num_features)
                    gn = nn.GroupNorm(groups, child.num_features)
                    setattr(module, name, gn)
                else:
                    self.replace_bn_with_gn(child, num_groups)

        def forward(self, x):
            """
            Forward logic.
            
            Input:
                x (torch.Tensor): Input image batch of shape (B, 1, H, W)
                
            Output:
                rotation_6d (torch.Tensor): Continuous rotation map (B, 6)
                translation (torch.Tensor): Translation vector (B, 3)
            """
            # Feature processing
            x = self.sobel(x)            # (B, 3, H, W)
            x = self.backbone(x)         # (B, 512, 1, 1)
            x = self.spatial_attn(x)

            # Dense layers
            x = self.mlp(x)            # (B, 9)
            rotation_6d = self.rotation(x)
            translation = self.translation(x)
            
            return rotation_6d, translation
elif choose == 3:
    class PoseRegressor(nn.Module):
        """
        Pose regressor mapping a single channel DRR image to a 6D continuous rotation
        and a 3D translation vector. Applies custom Sobel preprocess, ResNet18 backbone
        (re-initialized with GroupNorm), and an MLP prediction head.
        """
        def __init__(self, dropout=0.4, heads=4):
            super().__init__()
            self.sobel = SobelConv()
            
            # Base ResNet18 Native
            # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            resnet = models.resnet18(weights=None)
            
            # Replace BatchNorm with GroupNorm for smaller batch sizes stability
            self.replace_bn_with_gn(resnet)
            
            # Standard input for ResNet18 is 3 channels, so we just enforce initialization.
            # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            # nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # Extract features up to GAP output
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            
            with torch.no_grad():
                dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                feat = self.backbone(dummy)
                C, H, W = feat.shape[1:]  # skip batch dim 

            Q = 4
            CQ = C*Q
            self.attn = AttentionPool(dim=C, num_queries=Q, heads=heads)

            # MLP - shared layer + 6D rotation + translation
            self.mlp = nn.Sequential(
                nn.Linear(CQ, CQ//4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.rotation = nn.Sequential(
                nn.Linear(CQ//4, CQ//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(CQ//8, CQ//16),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(CQ//16, 6)  # 6D rotation
            )
            self.translation = nn.Sequential(
                nn.Linear(CQ//4, CQ//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(CQ//8, 3)  # 3D translation
            )

        def replace_bn_with_gn(self, module, num_groups=32):
            """
            Recursively replaces all nn.BatchNorm2d layers in a module with nn.GroupNorm.
            """
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    groups = min(num_groups, child.num_features)
                    gn = nn.GroupNorm(groups, child.num_features)
                    setattr(module, name, gn)
                else:
                    self.replace_bn_with_gn(child, num_groups)

        def forward(self, x):
            """
            Forward logic.
            
            Input:
                x (torch.Tensor): Input image batch of shape (B, 1, H, W)
                
            Output:
                rotation_6d (torch.Tensor): Continuous rotation map (B, 6)
                translation (torch.Tensor): Translation vector (B, 3)
            """
            # Feature processing
            x = self.sobel(x)           # (B, 3, H, W)
            x = self.backbone(x)        # (B, C, 8, 8)
            x = self.attn(x)            # (B, Q, C)

            x = x.reshape(x.shape[0], -1) # (B, Q*C)

            # Dense layers
            x = self.mlp(x)                     # (B, C/4)
            rotation_6d = self.rotation(x)      # (B, 6)
            translation = self.translation(x)   # (B, 3)
            
            return rotation_6d, translation
        
else:
    class PoseRegressor(nn.Module):
        """
        Pose regressor mapping a single channel DRR image to a 6D continuous rotation
        and a 3D translation vector. Applies custom Sobel preprocess, ResNet18 backbone
        (re-initialized with GroupNorm), and an MLP prediction head.
        """
        def __init__(self, dropout=0.3):
            super().__init__()
            self.sobel = SobelConv()
            
            # Base ResNet18 Native
            # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            resnet = models.resnet18(weights=None)
            
            # Replace BatchNorm with GroupNorm for smaller batch sizes stability
            self.replace_bn_with_gn(resnet)
            
            # Standard input for ResNet18 is 3 channels, so we just enforce initialization.
            # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            # nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # Extract features up to GAP output
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            
            with torch.no_grad():
                dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                feat = self.backbone(dummy)
                C, H, W = feat.shape[1:]  # skip batch dim 

            # Mamba over flattened spatial tokens with learned positional encoding
            self.positional_encoding = nn.Parameter(torch.zeros(H * W, C))
            torch.nn.init.trunc_normal_(self.positional_encoding, std=0.02)

            # directions = ["tl_row", "br_col", "tr_row", "bl_col", "br_row", "tl_col", "bl_row", "tr_col"]
            directions = ["tl_row", "br_row", "tr_col", "bl_col"]

            self.mamba = nn.ModuleList(
                [DirectionalMambaBlock(d_model=C, H=H, W=W, mode=m) for m in directions]
            )

            self.post_mamba = nn.Linear(512, 1)
                
            # Unified MLP, then 2 separate MLPs for translation, rotation
            self.mlp = nn.Sequential(
                nn.Linear(C, C//2),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            self.rotation = nn.Sequential(
                nn.Linear(C//2, C//4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//4, C//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//8, 6)  # 6D rotation + 3D translation
            )

            self.translation = nn.Sequential(
                nn.Linear(C//2, C//4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//4, C//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//8, 3)  # 6D rotation + 3D translation
            )

        def replace_bn_with_gn(self, module, num_groups=32):
            """
            Recursively replaces all nn.BatchNorm2d layers in a module with nn.GroupNorm.
            """
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    groups = min(num_groups, child.num_features)
                    gn = nn.GroupNorm(groups, child.num_features)
                    setattr(module, name, gn)
                else:
                    self.replace_bn_with_gn(child, num_groups)

        def forward(self, x):
            """
            Forward logic.
            
            Input:
                x (torch.Tensor): Input image batch of shape (B, 1, H, W)
                
            Output:
                rotation_6d (torch.Tensor): Continuous rotation map (B, 6)
                translation (torch.Tensor): Translation vector (B, 3)
            """
            # Feature processing
            x = self.sobel(x)            # (B, 3, H, W)
            x = self.backbone(x)         # (B, 512, 1, 1)
            
            B, C, H, W = x.shape
            tokens = x.view(B, C, H * W).permute(0, 2, 1) # (B, H*W, C)

            tokens = tokens + self.positional_encoding.unsqueeze(0) # (B, H*W, C)
            
            for block in self.mamba:
                tokens = block(tokens)

            # weights = torch.softmax(self.post_mamba(tokens), dim=1)  # (B, H*W, 1)
            # out = (tokens * weights).sum(dim=1)         # (B, C)
            out = tokens.mean(1) # (B, 1, C)

            # Dense layers
            out = self.mlp(out.squeeze(1)) # (B, 9)
            rotation_6d = self.rotation(out)
            translation = self.translation(out)
            
            return rotation_6d, translation
