import torch
import torch.nn as nn
import math
from mamba.mamba_ssm import Mamba

class Sobel(nn.Module):
    """
    Computes Sobel edge maps.
    Returns magnitude and optionally orientation.
    Resulting channels: 2 (if separate) or magnitude calculation.
    """
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(
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
        self.filter.weight = nn.Parameter(G, requires_grad=False)

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
        # We need to make sure perm is on the same device as tokens
        if self.perm.device != tokens.device:
             self.perm = self.perm.to(tokens.device)
             self.inv_perm = self.inv_perm.to(tokens.device)

        x = tokens.index_select(dim=1, index=self.perm)     # directional order
        x = self.block(x)                                   # Mamba along that order
        x = x.index_select(dim=1, index=self.inv_perm)      # back to row-major
        return x

class SpatialAttentionPool(nn.Module):
    def __init__(self, in_channels, num_queries=4):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_queries, 1)
        )

        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):  # (B, C, H, W)
        B, C, H, W = x.shape

        attn = self.attn(x).reshape(B, -1, H * W)   # (B, Q, HW)
        attn = torch.softmax(attn, dim=-1)

        x_flat = x.reshape(B, C, H * W)             # (B, C, HW)

        pooled = torch.bmm(attn, x_flat.transpose(1, 2))  # (B, Q, C)

        pooled = self.norm(pooled)

        return pooled.reshape(B, -1)  # (B, Q*C)
    
class AttentionPool(nn.Module):
    def __init__(self, dim=512, num_queries=4, heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, C)

        out, _ = self.attn(Q, x, x)  # (B, Q, C)
        out = self.proj(out)

        return out
    
class StructuredSpatialPool(nn.Module):
    """
    Splits the feature map into a 2x2 grid of regions and pools each region
    independently with both avg and max, then concatenates.
    Forces the model to reason about 4 distinct spatial regions.
    No learnable parameters → no collapse possible.
    Output: (B, C * 4 * 2) = (B, C * 8) for a 2x2 grid with avg+max
    """
    def __init__(self, grid=(2, 2)):
        super().__init__()
        self.grid = grid

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        gh, gw = self.grid
        cell_h, cell_w = H // gh, W // gw
        tokens = []
        for i in range(gh):
            for j in range(gw):
                region = x[:, :, i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                tokens.append(region.amax(dim=(-2, -1)))   # max pool
                tokens.append(region.mean(dim=(-2, -1)))   # avg pool
        return torch.cat(tokens, dim=1)  # (B, C * 8)
    