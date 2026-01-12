# vss_unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from layers import SALayer, CALayer, CurveCALayer

# pretty repr for debug
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# ---------------------------
# Helper conv
# ---------------------------
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride
    )


# ===========================
# SS2D: Visual State-Space 2D (VMamba core)
# ===========================
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

        # Projects channels-last features to (2 * d_inner) for gating (x,z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Depthwise conv for local context (expects channels-first input)
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

        # x_proj: 4 directional linear projections (we store weights as a parameter)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt_projs: per-direction small linear layers (stored as weights + bias param tensors)
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # State-space parameters
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True, device=device)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True, device=device)

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
        """
        Core selective-scan forward. `x` is expected as channels-first conv output: (B, C, H, W)
        Returns four directional outputs arranged for later combination.
        """
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # Build directional sequences: [h->w, w->h] and their flips -> total 4 sequences
        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, C', L)

        # Project to dt_rank + 2*d_state (per-direction)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        # Map small dt vectors to full d_inner via dt_projs
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        # Prepare arguments
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

        # Reconstruct directional outputs with proper flips/transposes
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Forward expects channels-last format: (B, H, W, C)
        Returns channels-last output of same (H, W, d_model)
        """
        # input shape (B, H, W, C)
        B, H, W, C = x.shape
        xz = self.in_proj(x)            # (B, H, W, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)      # x: (B,H,W,d_inner), z: gating (B,H,W,d_inner)

        # conv path expects (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))    # (B, d_inner, H, W)

        # forward core expects channels-first conv output
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32

        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)               # gating with z
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


# ===========================
# Dual Attention: Spatial + Curved (parallel)
# ===========================
class DualAttentionBlock(nn.Module):
    """
    Implements the dual attention structure: Spatial Attention (SALayer)
    and Curved Channel Attention (CurveCALayer) in parallel, fused by addition.
    Input: channels-first (B, C, H, W), Output: same shape.
    """
    def __init__(self, channel):
        super(DualAttentionBlock, self).__init__()
        self.conv1 = conv(channel, channel, kernel_size=3, bias=True)
        self.curved_attention = CurveCALayer(channel)
        self.spatial_attention = SALayer()
        self.conv2 = conv(channel, channel, kernel_size=3, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, H, W)
        x_conv1 = self.act(self.conv1(x))

        # parallel attention
        curved_out = self.curved_attention(x_conv1)   # (B,C,H,W)
        spatial_out = self.spatial_attention(x_conv1) # (B,C,H,W)

        # fuse (element-wise add)
        fused_attention = curved_out + spatial_out

        output = self.conv2(fused_attention)
        return output


# ===========================
# EnhancedVSSBlock (CAMamba-like module)
# ===========================
class EnhancedVSSBlock(nn.Module):
    """
    Channel-Aware Mamba (CAMamba) style block:
    - Upper branch: linear -> depthwise conv -> SS2D -> norm
    - Lower branch: linear gating path
    - Post: linear -> channel attention -> DualAttention -> residuals
    Expects channels-last input: (B, H, W, C)
    """
    def __init__(self, hidden_dim=64, d_state=16, d_conv=3):
        super(EnhancedVSSBlock, self).__init__()

        # initial norm (channels-last)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # upper Mamba branch
        self.linear_upper = nn.Linear(hidden_dim, hidden_dim)
        self.dw_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=hidden_dim,
            bias=True
        )
        self.ss2d = SS2D(d_model=hidden_dim, d_state=d_state)
        self.norm_upper = nn.LayerNorm(hidden_dim)

        # lower gating/skip branch
        self.linear_lower = nn.Linear(hidden_dim, hidden_dim)

        # post-add path
        self.linear_post = nn.Linear(hidden_dim, hidden_dim)
        self.norm_post = nn.LayerNorm(hidden_dim)
        self.channel_attention = CALayer(channel=hidden_dim)

        # dual attention block (channels-first)
        self.dual_attention = DualAttentionBlock(channel=hidden_dim)

        # final linear + norm
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.silu = nn.SiLU()

    def forward(self, x):
        """
        x: (B, H, W, C) channels-last
        returns: (B, H, W, C)
        """
        # preserve residual (channels-last)
        residual = x
        B, H, W, C = x.shape
        assert C == x.shape[-1], "Expected channels-last input"

        # initial norm
        x_norm = self.norm1(x)

        # Upper Mamba branch
        x_upper = self.linear_upper(x_norm)                     # (B,H,W,C)
        x_upper = x_upper.permute(0, 3, 1, 2).contiguous()      # (B,C,H,W)
        x_upper = self.dw_conv(x_upper)                         # (B,C,H,W)
        x_upper = self.silu(x_upper)
        x_upper = x_upper.permute(0, 2, 3, 1).contiguous()      # (B,H,W,C)
        x_upper = self.ss2d(x_upper)                            # SS2D expects channels-last
        x_upper = self.norm_upper(x_upper)

        # Lower gating branch
        x_lower = self.linear_lower(x_norm)
        x_lower = self.silu(x_lower)

        # Combine branches (element-wise)
        x_comb = x_upper + x_lower

        # Post-add path
        x_post = self.linear_post(x_comb)
        x_post = self.norm_post(x_post)

        # channel attention expects channels-first
        x_ca = x_post.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        x_ca = self.channel_attention(x_ca)
        x_ca = x_ca.permute(0, 2, 3, 1).contiguous()   # back to (B,H,W,C)

        # Residual add
        x = residual + x_ca

        # Dual attention (channels-first) with local residual
        x_ch_first = x.permute(0, 3, 1, 2).contiguous()   # (B,C,H,W)
        residual_attention = x_ch_first
        x_da = self.dual_attention(x_ch_first)
        x_da = x_da + residual_attention
        x_da = x_da.permute(0, 2, 3, 1).contiguous()      # (B,H,W,C)

        # final linear + norm
        x_final = self.linear2(x_da)
        x_final = self.ln2(x_final)

        # main residual
        out = residual + x_final
        return out


# ===========================
# Wrapper to integrate EnhancedVSSBlock into channel-first U-Net
# ===========================
class VSSBlockWrapper(nn.Module):
    """
    Wrap EnhancedVSSBlock to accept (B, C, H, W) and output (B, C_out, H, W)
    - Uses a 1x1 projection for residual channel matching
    - Internally converts to channels-last for EnhancedVSSBlock
    """
    def __init__(self, in_channels, out_channels, d_state=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.vss_block = EnhancedVSSBlock(hidden_dim=out_channels, d_state=d_state)

    def forward(self, x):
        # x: (B, C_in, H, W)
        residual = self.proj(x)  # (B, out_channels, H, W)

        # to channels-last (B, H, W, C)
        x_permuted = residual.permute(0, 2, 3, 1).contiguous()

        # pass through Enhanced VSS Block
        vss_out = self.vss_block(x_permuted)  # (B, H, W, C_out)

        # back to channels-first
        vss_out_permuted = vss_out.permute(0, 3, 1, 2).contiguous()  # (B, C_out, H, W)

        # the EnhancedVSSBlock already applies a residual; return the block output
        return vss_out_permuted


# ===========================
# Final VSS_UNet (lighter / memory-friendly)
# ===========================
class VSS_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(VSS_UNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder using VSSBlockWrapper
        self.encoder1 = VSSBlockWrapper(in_channels, 32)
        self.encoder2 = VSSBlockWrapper(32, 64)
        self.encoder3 = VSSBlockWrapper(64, 128)
        self.encoder4 = VSSBlockWrapper(128, 256)

        # Bottleneck
        self.bottleneck = VSSBlockWrapper(256, 512)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = VSSBlockWrapper(256 + 256, 256)  # after concat: 256 (up) + 256 (skip)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = VSSBlockWrapper(128 + 128, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = VSSBlockWrapper(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = VSSBlockWrapper(32 + 32, 32)

        # Final output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)               # -> (B,32,H,W)
        e2 = self.encoder2(self.pool(e1))   # -> (B,64,H/2,W/2)
        e3 = self.encoder3(self.pool(e2))   # -> (B,128,H/4,W/4)
        e4 = self.encoder4(self.pool(e3))   # -> (B,256,H/8,W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # -> (B,512,H/16,W/16)

        # Decoder stage 4
        d4 = self.upconv4(b)                # (B,256,H/8,W/8)
        d4 = torch.cat((d4, e4), dim=1)     # (B,512,...)
        d4 = self.decoder4(d4)              # -> (B,256,...)

        # Decoder stage 3
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        # Decoder stage 2
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        # Decoder stage 1
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        out = self.final_conv(d1)
        out = self.activation(out)
        return out


# If this script is run directly, quick smoke test (no external ops executed here)
if __name__ == "__main__":
    # smoke test shapes (random tensor)
    model = VSS_UNet(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print("Input:", x.shape, "Output:", y.shape)
