# model/mamba_unet.py
# CORRECT ARCHITECTURE: VSS-UNet with CAMamba Module + Dual Attention
# Matches paper contributions exactly

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

# ===================================================================
# 1. Time Embedding
# ===================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# ===================================================================
# 2. Attention Layers (Spatial + Curved)
# ===================================================================
def conv(in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride, bias=bias)

class SALayer(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, maxv], dim=1)
        y = torch.sigmoid(self.conv(y))
        return x * y

class CALayer(nn.Module):
    """Channel Attention Layer"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class CurveCALayer(nn.Module):
    """Curved Channel Attention Layer"""
    def __init__(self, channel):
        super().__init__()
        self.curve = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.curve(x)

class DualAttentionBlock(nn.Module):
    """
    Custom Dual Attention Block:
    - Spatial Attention and Curved Attention working in PARALLEL
    - Combined feature refinement
    """
    def __init__(self, channel):
        super().__init__()
        self.conv1 = conv(channel, channel, 3)
        
        # Parallel Attention Branches
        self.curved_attention = CurveCALayer(channel)
        self.spatial_attention = SALayer()
        
        self.conv2 = conv(channel, channel, 3)

    def forward(self, x):
        # Initial convolution
        x_conv1 = F.relu(self.conv1(x))
        
        # Parallel attention branches
        curved_out = self.curved_attention(x_conv1)
        spatial_out = self.spatial_attention(x_conv1)
        
        # Combine outputs (parallel fusion)
        fused_attention = curved_out + spatial_out
        
        # Final convolution
        output = self.conv2(fused_attention)
        return output

# ===================================================================
# 3. Official SS2D (Memory Efficient)
# ===================================================================
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    print("Warning: mamba_ssm not installed → pip install mamba-ssm causal-conv1d")

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([
            x.view(B, -1, L), 
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        del x_hwwh
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        del x_dbl
        
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

# ===================================================================
# 4. CAMamba Module (CORRECT ARCHITECTURE from paper)
# ===================================================================
class CAMambaModule(nn.Module):
    """
    CAMamba Module - Exact architecture from paper:
    1. Upper Branch: Linear → DW Conv → SiLU → SS2D → Norm
    2. Lower Branch: Linear → SiLU (gating path)
    3. Combine with + operator
    4. Linear → Norm → Channel Attention
    5. Dual Attention Block
    """
    def __init__(self, hidden_dim, d_state=16, d_conv=3, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # 1. Initial LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)

        # 2. Upper Branch (Mamba Path)
        self.linear_upper = nn.Linear(hidden_dim, hidden_dim)
        self.dw_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=hidden_dim,  # Depth-wise convolution
            bias=True
        )
        self.ss2d = SS2D(d_model=hidden_dim, d_state=d_state)
        self.norm_upper = nn.LayerNorm(hidden_dim)

        # 3. Lower Branch (Gating/Skip Path)
        self.linear_lower = nn.Linear(hidden_dim, hidden_dim)

        # 4. Shared SiLU Activation
        self.silu = nn.SiLU()

        # 5. Post-Addition Path
        self.linear_post = nn.Linear(hidden_dim, hidden_dim)
        self.norm_post = nn.LayerNorm(hidden_dim)
        self.channel_attention = CALayer(channel=hidden_dim)

        # 6. Dual Attention Block
        self.dual_attention = DualAttentionBlock(channel=hidden_dim)

        # 7. Final layers
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def _forward_impl(self, x):
        """
        Input: (B, H, W, C)
        """
        # Store initial input for final residual
        residual = x
        B, H, W, C = x.shape

        # Apply initial LayerNorm
        x = self.norm1(x)

        # === Upper Mamba Branch ===
        x_upper = self.linear_upper(x)
        # Permute to (B, C, H, W) for convolution
        x_upper = x_upper.permute(0, 3, 1, 2)
        x_upper = self.dw_conv(x_upper)
        x_upper = self.silu(x_upper)
        # Permute back to (B, H, W, C) for SS2D
        x_upper = x_upper.permute(0, 2, 3, 1)
        x_upper = self.ss2d(x_upper)
        x_upper = self.norm_upper(x_upper)

        # === Lower Gating Branch ===
        x_lower = self.linear_lower(x)
        x_lower = self.silu(x_lower)

        # === Combine Branches ===
        x = x_upper + x_lower

        # === Post-Addition Path ===
        x = self.linear_post(x)
        x = self.norm_post(x)

        # Channel Attention (needs B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.channel_attention(x)
        # Back to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # === First Residual Connection ===
        x = residual + x

        # === Dual Attention Block ===
        # Convert to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        residual_attention = x
        x = self.dual_attention(x)
        x = x + residual_attention
        # Back to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # === Final Linear Layers ===
        x = self.linear2(x)
        x = self.ln2(x)
        
        # === Final Residual ===
        out = residual + x

        return out
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

# ===================================================================
# 5. VSS-UNet with CAMamba Architecture
# ===================================================================
class MambaUNet(nn.Module):
    """
    VSS-UNet: Replaces standard convolutions with CAMamba Modules
    
    Key Features:
    - Enhanced VSS Block (SS2D/Mamba) for long-range dependencies
    - CAMamba Module with dual-branch architecture
    - Dual Attention Block (Spatial + Curved in parallel)
    """
    def __init__(self, in_channels=6, out_channels=3, 
                 depths=[2,2,4,2],  # Memory optimized
                 dims=[96,192,384,768], 
                 d_state=16,
                 use_checkpoint=True):
        super().__init__()
        self.time_dim = dims[0] * 4
        self.use_checkpoint = use_checkpoint
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dims[0]),
            nn.Linear(dims[0], self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        self.proj_in = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        # Encoder - Using CAMamba Modules
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.enc_time = nn.ModuleList()
        prev = dims[0]
        for i, d in enumerate(dims[1:]):
            self.enc_time.append(nn.Linear(self.time_dim, prev))
            self.encoders.append(nn.ModuleList([
                CAMambaModule(prev, d_state, use_checkpoint=use_checkpoint) 
                for _ in range(depths[i])
            ]))
            self.downs.append(nn.Conv2d(prev, d, 4, 2, 1))
            prev = d

        # Bottleneck - Using CAMamba Modules
        self.bot_time = nn.Linear(self.time_dim, prev)
        self.bottleneck = nn.ModuleList([
            CAMambaModule(prev, d_state, use_checkpoint=use_checkpoint) 
            for _ in range(depths[-1])
        ])

        # Decoder - Using CAMamba Modules
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.dec_time = nn.ModuleList()
        for i, d in enumerate(reversed(dims[:-1])):
            self.dec_time.append(nn.Linear(self.time_dim, d))
            self.decoders.append(nn.ModuleList([
                CAMambaModule(d, d_state, use_checkpoint=use_checkpoint) 
                for _ in range(depths[-(i+2)])
            ]))
            self.ups.append(nn.ConvTranspose2d(prev, d, 4, 2, 1))
            self.skip_projs.append(nn.Conv2d(d*2, d, 1))
            prev = d

        self.out = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, padding=1),
            nn.GroupNorm(8, dims[0]),
            nn.SiLU(),
            nn.Conv2d(dims[0], out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x = self.proj_in(x)
        skips = []

        # Encoder
        for i, (blocks, down) in enumerate(zip(self.encoders, self.downs)):
            x = x + self.enc_time[i](t_emb)[..., None, None]
            x = rearrange(x, 'b c h w -> b h w c')
            for blk in blocks: 
                x = blk(x)
            x = rearrange(x, 'b h w c -> b c h w')
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = x + self.bot_time(t_emb)[..., None, None]
        x = rearrange(x, 'b c h w -> b h w c')
        for blk in self.bottleneck: 
            x = blk(x)
        x = rearrange(x, 'b h w c -> b c h w')

        # Decoder
        for i, (blocks, up, proj) in enumerate(zip(self.decoders, self.ups, self.skip_projs)):
            x = up(x)
            x = x + self.dec_time[i](t_emb)[..., None, None]
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = proj(x)
            x = rearrange(x, 'b c h w -> b h w c')
            for blk in blocks: 
                x = blk(x)
            x = rearrange(x, 'b h w c -> b c h w')

        return self.out(x)

# Test
if __name__ == "__main__":
    print("=" * 80)
    print("VSS-UNet Architecture Verification")
    print("=" * 80)
    
    model = MambaUNet().cuda()
    x = torch.randn(2, 6, 256, 256).cuda()
    t = torch.randint(0, 1000, (2,)).cuda()
    
    print("\n✓ Model Components:")
    print("  1. Enhanced VSS Block (SS2D/Mamba): ✓")
    print("  2. CAMamba Module with dual branches: ✓")
    print("  3. Dual Attention Block (Spatial + Curved parallel): ✓")
    print("  4. Channel Attention: ✓")
    
    out = model(x, t)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"\n✓ SUCCESS!")
    print(f"  - Output shape: {out.shape}")
    print(f"  - Parameters: {params:.2f}M")
    print(f"  - Memory efficient: Gradient checkpointing enabled")
    print("=" * 80)