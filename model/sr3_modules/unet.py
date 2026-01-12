import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

# === WINDOW UTILS ===
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# === SWIN BLOCK ===
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward_impl(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: L={L}, H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, window_size*window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, x_windows, x_windows, need_weights=False)[0]  # (nW*B, window_size*window_size, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # (B, Hp, Wp, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Unpad
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward(self, x, H, W):
        # Use gradient checkpointing if in training mode
        if self.training:
            return checkpoint(self.forward_impl, x, H, W, use_reentrant=False)
        else:
            return self.forward_impl(x, H, W)

# === HYBRID U-NET ===
class HybridUNet(nn.Module):
    def __init__(self, in_channel=6, out_channel=3, inner_channel=64, channel_mults=[1,2,4,8],
                 res_blocks=3, dropout=0.05, num_heads=8, window_size=8):
        super().__init__()
        self.inner_channel = inner_channel
        self.channel_mults = channel_mults
        self.res_blocks = res_blocks
        self.window_size = window_size

        self.proj_in = nn.Conv2d(in_channel, inner_channel, 3, padding=1)

        # Encoder channels
        self.enc_channels = [inner_channel * m for m in channel_mults]

        # === ENCODER ===
        self.enc_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i, out_ch in enumerate(self.enc_channels):
            in_ch = inner_channel if i == 0 else self.enc_channels[i-1]
            for j in range(res_blocks):
                # Alternate between regular and shifted window attention
                shift = 0 if j % 2 == 0 else window_size // 2
                self.enc_blocks.append(
                    SwinBlock(out_ch, num_heads=num_heads, window_size=window_size, 
                             shift_size=shift, drop=dropout)
                )
            if i < len(self.enc_channels) - 1:
                self.downsample.append(nn.Conv2d(out_ch, self.enc_channels[i+1], 3, 2, 1))

        # === BOTTLENECK ===
        self.mid_block = nn.Sequential(
            SwinBlock(self.enc_channels[-1], num_heads=num_heads*2, window_size=window_size, 
                     shift_size=0, drop=dropout),
            SwinBlock(self.enc_channels[-1], num_heads=num_heads*2, window_size=window_size, 
                     shift_size=window_size//2, drop=dropout),
        )

        # === DECODER ===
        self.dec_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.skip_proj = nn.ModuleList()

        for i in reversed(range(len(self.enc_channels))):
            curr_ch = self.enc_channels[i]
            
            for j in range(res_blocks + 1):
                shift = 0 if j % 2 == 0 else window_size // 2
                self.dec_blocks.append(
                    SwinBlock(curr_ch, num_heads=num_heads, window_size=window_size, 
                             shift_size=shift, drop=dropout)
                )

            if i > 0:
                self.upsample.append(nn.ConvTranspose2d(curr_ch, self.enc_channels[i-1], 3, 2, 1, 1))
            
            # Skip projection: from concatenated channels to current channel
            if i < len(self.enc_channels) - 1:
                skip_ch = self.enc_channels[i]
                self.skip_proj.append(nn.Conv2d(skip_ch * 2, curr_ch, 1))

        # Add residual connection from input
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.enc_channels[0], self.enc_channels[0], 3, padding=1),
            nn.GroupNorm(8, self.enc_channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.enc_channels[0], out_channel, 3, padding=1)
        )
        
        self.out = self.final_conv

    def forward(self, x, t=None):
        x = self.proj_in(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        
        # Clear cache before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        x_seq = rearrange(x, 'b c h w -> b (h w) c')

        skips = []
        skip_sizes = []
        current_H, current_W = H, W

        # === ENCODER ===
        block_idx = 0
        for i in range(len(self.enc_channels)):
            for _ in range(self.res_blocks):
                x_seq = self.enc_blocks[block_idx](x_seq, current_H, current_W)
                block_idx += 1
            
            # Store skip BEFORE downsampling
            if i < len(self.downsample):
                # Detach skip to save memory
                skips.append(x_seq.detach())
                skip_sizes.append((current_H, current_W))
                x_seq = rearrange(x_seq, 'b (h w) c -> b c h w', h=current_H, w=current_W)
                x_seq = self.downsample[i](x_seq)
                current_H, current_W = x_seq.shape[2], x_seq.shape[3]
                x_seq = rearrange(x_seq, 'b c h w -> b (h w) c')

        # === BOTTLENECK ===
        x_seq = self.mid_block[0](x_seq, current_H, current_W)
        x_seq = self.mid_block[1](x_seq, current_H, current_W)

        # === DECODER ===
        block_idx = 0
        proj_idx = 0
        
        for i in reversed(range(len(self.enc_channels))):
            # Upsample first (except at deepest level)
            if i < len(self.enc_channels) - 1:
                x_seq = rearrange(x_seq, 'b (h w) c -> b c h w', h=current_H, w=current_W)
                x_seq = self.upsample[len(self.enc_channels) - 2 - i](x_seq)
                current_H, current_W = x_seq.shape[2], x_seq.shape[3]
                
                # Add skip connection
                skip = skips.pop()
                skip_H, skip_W = skip_sizes.pop()
                
                # Re-enable gradient for skip
                skip = skip.requires_grad_(True)
                skip = rearrange(skip, 'b (h w) c -> b c h w', h=skip_H, w=skip_W)
                
                # Concatenate and project
                x_seq = torch.cat([x_seq, skip], dim=1)
                x_seq = self.skip_proj[proj_idx](x_seq)
                proj_idx += 1
                
                x_seq = rearrange(x_seq, 'b c h w -> b (h w) c')
                
                # Clear cache after skip connection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Apply decoder blocks
            for _ in range(self.res_blocks + 1):
                x_seq = self.dec_blocks[block_idx](x_seq, current_H, current_W)
                block_idx += 1

        x_out = rearrange(x_seq, 'b (h w) c -> b c h w', h=current_H, w=current_W)
        return self.out(x_out)