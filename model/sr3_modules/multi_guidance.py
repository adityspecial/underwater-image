import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicalGuidance(nn.Module):
    """
    Physical model guidance for underwater image enhancement
    Measures color cast and haze effects
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x_restored, x_degraded):
        """
        Args:
            x_restored: Enhanced image (B, 3, H, W) in range [-1, 1]
            x_degraded: Input degraded image (B, 3, H, W) in range [-1, 1]
        """
        # Convert to [0, 1] range
        x_restored = (x_restored + 1) / 2
        x_degraded = (x_degraded + 1) / 2
        
        # Color cast loss: penalize green/blue dominance (underwater characteristic)
        r, g, b = x_restored[:, 0], x_restored[:, 1], x_restored[:, 2]
        color_cast = torch.mean((g - r).abs() + (b - r).abs())
        
        # Dark channel prior for haze
        dark_channel = torch.min(x_restored, dim=1)[0]
        haze_loss = torch.mean(dark_channel)
        
        return 0.5 * color_cast + 0.5 * haze_loss


class EdgeGuidance(nn.Module):
    """
    Edge preservation guidance
    Encourages sharp edges in enhanced images
    """
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
    
    def forward(self, x):
        """
        Args:
            x: Image (B, 3, H, W) in range [-1, 1]
        """
        # Convert to [0, 1] and grayscale
        x = (x + 1) / 2
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Compute gradients
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # Edge magnitude
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        # Encourage edges (higher magnitude = sharper)
        # Negative because we want to maximize edges
        return -torch.mean(edge_mag)


class IQAGuidance(nn.Module):
    """
    Image Quality Assessment guidance
    Based on simple quality metrics
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: Image (B, 3, H, W) in range [-1, 1]
        """
        # Convert to [0, 1]
        x = (x + 1) / 2
        
        # Brightness variance (penalize over/under exposure)
        brightness = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        brightness_var = torch.var(brightness, dim=[1, 2])
        brightness_loss = torch.mean((brightness_var - 0.1).abs())
        
        # Color saturation
        max_rgb = torch.max(x, dim=1)[0]
        min_rgb = torch.min(x, dim=1)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        saturation_loss = -torch.mean(saturation)  # Encourage saturation
        
        # Contrast (standard deviation)
        contrast = torch.std(x, dim=[2, 3])
        contrast_loss = -torch.mean(contrast)  # Encourage contrast
        
        return brightness_loss + 0.5 * saturation_loss + 0.3 * contrast_loss