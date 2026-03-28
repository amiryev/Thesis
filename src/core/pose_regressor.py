import torch
import torch.nn as nn
import torchvision.models as models

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

    def forward(self, x):
        """
        Input: 
            x (torch.Tensor): Tensor of shape (B, 1, H, W).
        Output:
            (torch.Tensor): Tensor of shape (B, 3, H, W) [raw, magnitude, orientation].
        """
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-6)
        orientation = torch.atan2(gy, gx)
        
        return torch.cat([x, magnitude, orientation], dim=1)

def replace_bn_with_gn(module, num_groups=32):
    """
    Recursively replaces all nn.BatchNorm2d layers in a module with nn.GroupNorm.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = min(num_groups, child.num_features)
            gn = nn.GroupNorm(groups, child.num_features)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)

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
        
        # Replace BatchNorm with GroupNorm for smaller batch sizes stability
        # replace_bn_with_gn(resnet)
        
        # Standard input for ResNet18 is 3 channels, so we just enforce initialization.
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
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
