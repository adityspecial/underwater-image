# mamba_hybrid_unet.py — FIXED VERSION (Time Embedding Bug Resolved)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
    print("✓ Mamba-SSM loaded successfully!")
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠ Mamba-SSM not available - using Conv-only fallback")

# ==================== TIME EMBEDDING ====================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        x = x.float()
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ==================== FIXED SS2D MODULE ====================
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = d_model // 16

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv//2, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()

        # Fixed parameter initialization
        self.x_proj = nn.Parameter(torch.randn(4, self.dt_rank + self.d_state * 2, self.d_inner) * 0.02)
        self.dt_projs_weight = nn.Parameter(torch.randn(4, self.d_inner, self.dt_rank) * 0.02)
        self.dt_projs_bias = nn.Parameter(torch.randn(4, self.d_inner) * 0.02)

        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A))
        self.Ds = nn.Parameter(torch.ones(4 * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # (B, H, W, C)
        B, H, W, C = x.shape
        L = H * W

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2)  # (B, d_inner, H, W)
        x = self.act(self.conv2d(x)).permute(0, 2, 3, 1)  # (B, H, W, d_inner)

        if not MAMBA_AVAILABLE:
            return self.out_proj(x.mean(dim=(1, 2), keepdim=True).expand(-1, H, W, -1))

        # 4-directional scan
        x_h = x
        x_w = x.transpose(1, 2)
        xs = torch.stack([x_h, x_w, x_h.flip(-1), x_w.flip(-1)], dim=1)  # (B, 4, H, W, C)
        xs = xs.view(B, 4, self.d_inner, L).float()

        # Project to dt, B, C
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight) + self.dt_projs_bias.unsqueeze(-1)

        # Selective scan
        ys = selective_scan_fn(
            xs, dts,
            -torch.exp(self.A_logs.float()).repeat(4, 1, 1),
            Bs, Cs,
            self.Ds.float().view(4, self.d_inner),
            delta_bias=self.dt_projs_bias.flatten(),
            delta_softplus=True
        )

        # Merge directions
        y = (ys[:, 0] + ys[:, 2].flip(-1) + 
             ys[:, 1].transpose(1, 2) + ys[:, 3].transpose(1, 2).flip(-1))
        y = y.view(B, H, W, -1)

        y = self.out_norm(y) * F.silu(z)
        return self.out_proj(self.dropout(y))

# ==================== MAMBA BLOCK ====================
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = SS2D(dim, d_state=d_state, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):  # (B, H, W, C)
        x = x + self.drop_path(self.mamba(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ==================== HYBRID BLOCK ====================
class HybridBlock(nn.Module):
    def __init__(self, dim, d_state=16, use_mamba=True):
        super().__init__()
        self.use_mamba = use_mamba and MAMBA_AVAILABLE
        
        if self.use_mamba:
            self.mamba_block = MambaBlock(dim, d_state=d_state)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
        )

    def forward(self, x):  # (B, C, H, W)
        residual = x
        
        if self.use_mamba:
            x_mamba = rearrange(x, 'b c h w -> b h w c')
            x_mamba = self.mamba_block(x_mamba)
            x_mamba = rearrange(x_mamba, 'b h w c -> b c h w')
        else:
            x_mamba = x
        
        x_conv = self.conv_block(x)
        return F.silu(residual + x_mamba + x_conv)

# ==================== MAIN U-NET (FIXED) ====================
class MambaEnhancedUNet(nn.Module):
    def __init__(self, in_channel=6, out_channel=3, inner_channel=64, 
                 channel_mults=[1, 2, 4, 8], res_blocks=2, dropout=0.0):
        super().__init__()
        self.inner_channel = inner_channel
        self.channels = [inner_channel * m for m in [1] + channel_mults]

        # Time embedding (FIXED)
        time_dim = inner_channel * 4
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(inner_channel),
            nn.Linear(inner_channel, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.time_proj = nn.ModuleList([
            nn.Linear(time_dim, ch) for ch in self.channels
        ])

        # Input
        self.input_conv = nn.Conv2d(in_channel, self.channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        for i in range(len(channel_mults)):
            blocks = nn.ModuleList([
                HybridBlock(self.channels[i+1]) for _ in range(res_blocks)
            ])
            self.down_blocks.append(nn.ModuleList([
                nn.Conv2d(self.channels[i], self.channels[i+1], 3, stride=2, padding=1),
                blocks
            ]))

        # Bottleneck
        self.mid_blocks = nn.ModuleList([
            HybridBlock(self.channels[-1]) for _ in range(2)
        ])

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channel_mults))):
            blocks = nn.ModuleList([
                HybridBlock(self.channels[i+1]) for _ in range(res_blocks + 1)
            ])
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose2d(self.channels[i+1], self.channels[i], 4, stride=2, padding=1),
                blocks,
                nn.Conv2d(self.channels[i+1]*2, self.channels[i], 1)
            ]))

        # Output
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[0], 3, padding=1),
            nn.GroupNorm(8, self.channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.channels[0], out_channel, 3, padding=1)
        )

    def forward(self, x, t=None):  # FIXED TIME EMBEDDING
        B = x.shape[0]

        # Time embedding (FIXED - broadcast correctly)
        if t is not None:
            t_emb = self.time_emb(t)  # (B, time_dim)
        else:
            t_emb = torch.zeros(B, self.time_emb[-1].out_features, device=x.device)

        # Input
        x = self.input_conv(x)
        x = x + self.time_proj[0](t_emb).view(B, -1, 1, 1)  # FIXED

        # Encoder
        skips = []
        for i, (down, blocks) in enumerate(self.down_blocks):
            x = down(x)
            x = x + self.time_proj[i+1](t_emb).view(B, -1, 1, 1)  # FIXED
            for block in blocks:
                x = block(x)
            skips.append(x)

        # Bottleneck
        x = x + self.time_proj[-1](t_emb).view(B, -1, 1, 1)  # FIXED
        for block in self.mid_blocks:
            x = block(x)

        # Decoder
        for i, (up, blocks, skip_proj) in enumerate(self.up_blocks):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = skip_proj(x)
            x = x + self.time_proj[len(self.channels)-2-i](t_emb).view(B, -1, 1, 1)  # FIXED
            for block in blocks:
                x = block(x)

        return self.final_conv(x)

# ==================== TESTING ====================
if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaEnhancedUNet(
        in_channel=6,
        out_channel=3,
        inner_channel=64,
        channel_mults=[1, 2, 4, 8],
        res_blocks=2,
        use_mamba=True
    ).to(device)
    
    # Test input: [condition + noisy_image]
    x = torch.randn(2, 6, 256, 256).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"✓ Model output shape: {out.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"✓ Mamba enabled: {model.use_mamba}")
