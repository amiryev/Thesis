import torch
import torch.nn as nn
import torch.functional as F
import torchvision.io as io
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import resnet18, ResNet18_Weights

from src.core.layers import SpatialAttentionPool, DirectionalMambaBlock, AttentionPool, StructuredSpatialPool
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

choose = "xrayencoder"
if choose == "basic":
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
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # resnet = models.resnet18(weights=None)
            
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

            # MLP - shared layer + 6D rotation + translation
            self.mlp = nn.Sequential(
                nn.Linear(C, C//4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.rotation = nn.Sequential(
                nn.Linear(C//4, C//16),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//16, 6)  # 6D rotation + 3D translation
            )
            self.translation = nn.Sequential(
                nn.Linear(C//4, C//16),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//16, 3)  # 6D rotation + 3D translation
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
            
            rotation_6d = self.rotation(out)      # (B, 6)
            translation = self.translation(out)   # (B, 3)
            
            return rotation_6d, translation
elif choose == "spatialAttention":
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
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # resnet = models.resnet18(weights=None)
            
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
elif choose == "attention":
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
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # resnet = models.resnet18(weights=None)
            
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
                    # groups = min(num_groups, child.num_features)
                    groups = min(num_groups, child.num_features // 2)
                    groups = max(1, groups)  # prevent groups > channels
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
elif choose == "mamba":
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
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # resnet = models.resnet18(weights=None)
            
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
elif choose == "avgMaxPool":
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
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # resnet = models.resnet18(weights=None)
            
            # Replace BatchNorm with GroupNorm for smaller batch sizes stability
            self.replace_bn_with_gn(resnet)
            
            # Standard input for ResNet18 is 3 channels, so we just enforce initialization.
            # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            # nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # Extract features up to GAP output
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            # layers_to_freeze = [
            #     self.backbone[0],  # conv1
            #     self.backbone[1],  # bn1/gn1
            #     self.backbone[4],  # layer1
            #     self.backbone[5],  # layer2
            # ]
            # for layer in layers_to_freeze:
            #     for param in layer.parameters():
            #         param.requires_grad = False

            with torch.no_grad():
                dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                feat = self.backbone(dummy)
                C, H, W = feat.shape[1:]  # skip batch dim 

            # Add feature map dropout
            self.feature_dropout = nn.Dropout2d(p=0.1)

            # MLP - shared layer + 6D rotation + translation
            self.mlp = nn.Sequential(
                nn.Linear(2*C, C//2),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.rotation = nn.Linear(C//2, 6)
            self.translation = nn.Linear(C//2, 3)

        def replace_bn_with_gn(self, module, num_groups=32):
            """
            Recursively replaces all nn.BatchNorm2d layers in a module with nn.GroupNorm.
            """
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    # groups = min(num_groups, child.num_features)
                    groups = min(num_groups, child.num_features // 2)
                    groups = max(1, groups)  # prevent groups > channels
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
            x = self.feature_dropout(x) # randomly zero entire channels

            # Apply GAP and GMP to reduce feature maps
            gap = x.mean(dim=(-2, -1))              # (B, C)
            gmp = x.amax(dim=(-2, -1))              # (B, C)
            pooled = torch.cat([gap, gmp], dim=1)   # (B, 2C)

            # Dense layers
            out = self.mlp(pooled)                # (B, C/2)
            rotation_6d = self.rotation(out)      # (B, 6)
            translation = self.translation(out)   # (B, 3)
            
            return rotation_6d, translation
elif choose == "spatialPool":
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
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # resnet = models.resnet18(weights=None)
            
            # Replace BatchNorm with GroupNorm for smaller batch sizes stability
            self.replace_bn_with_gn(resnet)
            
            # Standard input for ResNet18 is 3 channels, so we just enforce initialization.
            # resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            # nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
            
            # Extract features up to GAP output
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            # layers_to_freeze = [
            #     self.backbone[0],  # conv1
            #     self.backbone[1],  # bn1/gn1
            #     self.backbone[4],  # layer1
            #     self.backbone[5],  # layer2
            # ]
            # for layer in layers_to_freeze:
            #     for param in layer.parameters():
            #         param.requires_grad = False

            with torch.no_grad():
                dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                feat = self.backbone(dummy)
                C, H, W = feat.shape[1:]  # skip batch dim 

            # Add feature map dropout
            # self.feature_dropout = nn.Dropout2d(p=0.1)

            self.pooling = StructuredSpatialPool()

            # MLP - shared layer + 6D rotation + translation
            self.mlp = nn.Sequential(
                nn.Linear(8 * C, C),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C, C//2),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            # self.rotation = nn.Linear(C//2, 6)
            # self.translation = nn.Linear(C//2, 3)
            self.rotation = nn.Sequential(
                nn.Linear(C//2, C//8),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(C//8, 6)  # 6D rotation + 3D translation
            )
            self.translation = nn.Sequential(
                nn.Linear(C//2, C//8),
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
                    # groups = min(num_groups, child.num_features)
                    groups = min(num_groups, child.num_features // 2)
                    groups = max(1, groups)  # prevent groups > channels
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
            # x = self.feature_dropout(x) # randomly zero entire channels

            x = self.pooling(x)

            # Dense layers
            out = self.mlp(x)                # (B, C/2)
            rotation_6d = self.rotation(out)      # (B, 6)
            translation = self.translation(out)   # (B, 3)
            
            return rotation_6d, translation
elif choose == "xrayencoder":
    class PoseRegressor(nn.Module):
        """
        Masked reconstruction autoencoder:
        input: x (B,1,H,W)   # raw projection in [0,1]
        masking: patch-wise mask on x (zeros on masked patches)
        backbone: concat [x_masked, Sobel mag, Sobel orient] -> VGG16.features -> Mamba
        decoder: upsample to (B,1,H,W)
        output: recon (B,1,H,W), pixel_mask (B,1,H,W), features
        """
        def __init__(self, dropout: int = 0.3, device=config.DEVICE, size: int = config.IMAGE_SIZE, patch_size: int = config.PATCH_SIZE):
            super().__init__()
            self.device = device
            self.size = size
            self.ps = patch_size

            assert size % patch_size == 0, "size must be divisible by patch_size"
            self.num_patches_h = size // patch_size
            self.num_patches_w = size // patch_size
            self.num_patches = self.num_patches_h * self.num_patches_w

            # Inputs: (x_masked, sobel_mag, sobel_orient) -> 3 channels to match VGG
            # self.sobel = Sobel()
            self.sobel = SobelConv()

            # Encoder backbone
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)  # (B, 512, 4, 4) for 128x128
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            # self.backbone = vgg16(weights=VGG16_Weights.DEFAULT).features  # (B, 512, 4, 4) for 128x128

            with torch.no_grad():
                dummy = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                feat = self.backbone(dummy)
                C, H, W = feat.shape[1:]  # skip batch dim 

            # Mamba over flattened spatial tokens with learned positional encoding
            self.positional_encoding = nn.Parameter(torch.zeros(H * W, C))
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

            self.mlp = nn.ModuleList(
                [DirectionalMambaBlock(d_model=C, H=H, W=W, mode=m) for m in directions]
            )

            self.rotation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),  # (B,256,6,6)
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),  # (B,128,4,4)
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),   # (B,64,2,2)
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),   # (B,64,1,1)
                nn.ReLU(),
                nn.Flatten(),                                             # (B,64)
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 6),
            )

            self.translation = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),  # (B,256,6,6)
                nn.ReLU(),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),  # (B,128,4,4)
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),   # (B,64,2,2)
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),   # (B,64,1,1)
                nn.ReLU(),
                nn.Flatten(),                                             # (B,64)
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )


        @torch.no_grad()
        def load_kernel(self, crm_path, border=3):
            crm = (io.read_image(crm_path).float().to("cuda") / 255.0).unsqueeze(0)
            crm = F.interpolate(crm, size=(self.size, self.size), mode='bilinear', align_corners=False)
            kernel = (crm != 0).float()

            if border > 0:
                kernel_in = F.max_pool2d(1.0 - kernel.float(), kernel_size=2 * border + 1, stride=1, padding=border,)
                kernel_valid = (kernel_in == 0)       
            else:
                kernel_valid = kernel.bool()      

            return kernel_valid.detach()


        def _make_patch_mask(self, B: int, mask_ratio: float):
            """Return patch-level mask as expanded pixel mask (B,1,H,W)."""
            if mask_ratio == 0.0:
                return None
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

        def encode(self, x: torch.Tensor, kernel=None, patch_mask: torch.Tensor = None):
            # mag, orient = self.sobel(x, return_orientation=True)  # (B,1,H,W) each
            x3 = self.sobel(x)  # (B,1,H,W) each
            # if kernel is not None:
            #     x3 = torch.cat([x, mag, orient], dim=1) * kernel
            # else:              # (B,3,H,W)
            #     x3 = torch.cat([x, mag, orient], dim=1)
            x = self.backbone(x3)  # (B,512,4,4)

            B, C, H, W = x.shape
            tokens = x.view(B, C, H * W).permute(0, 2, 1)  # (B,L,C)

            if patch_mask is not None:
                mask = patch_mask.bool()
                tokens[mask] = self.mask_token
            
            tokens = tokens + self.positional_encoding.unsqueeze(0)  # (B,L,C)
            
            for block in self.mlp:
                tokens = block(tokens)

            tokens = tokens.permute(0, 2, 1).contiguous().view(B, C, H, W)
            return tokens

        def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
            """
            x: (B,1,H,W) in [0,1]
            returns:
            recon: (B,1,H,W)
            pixel_mask: (B,1,H,W) with 1 where masked
            features: encoder features before decoder (B,512,4,4)
            """
            B, C, H, W = x.shape
            assert C == 1, "Expect (B,1,H,W) grayscale input"

            # _, patch_mask = self._make_patch_mask(B, mask_ratio)
            patch_mask = None

            feats = self.encode(x, patch_mask=patch_mask)

            rot = self.rotation(feats)
            trans = self.translation(feats)
            pose = torch.cat([rot, trans], dim=-1)
            
            return rot, trans
