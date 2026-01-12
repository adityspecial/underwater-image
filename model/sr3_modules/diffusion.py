# model/sr3_modules/diffusion.py — OPTIMIZED FOR ENHANCED U-NET + HIGH PSNR
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import logging

logger = logging.getLogger('base')


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    linear_start = float(linear_start)
    linear_end = float(linear_end)
    cosine_s = float(cosine_s)

    if schedule == "linear":
        return np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        return betas.clamp(max=0.999).numpy()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# === INLINE GUIDANCE MODULES ===
class PhysicalGuidance(nn.Module):
    """Physical model guidance for underwater image enhancement"""
    def __init__(self):
        super().__init__()

    def forward(self, x_restored, x_degraded):
        x_restored = (x_restored + 1) / 2
        x_degraded = (x_degraded + 1) / 2
        r, g, b = x_restored[:, 0], x_restored[:, 1], x_restored[:, 2]
        color_cast = torch.mean((g - r).abs() + (b - r).abs())
        dark_channel = torch.min(x_restored, dim=1)[0]
        haze_loss = torch.mean(dark_channel)
        return 0.5 * color_cast + 0.5 * haze_loss


class EdgeGuidance(nn.Module):
    """Edge preservation guidance"""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x):
        x = (x + 1) / 2
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return -torch.mean(edge_mag)


class IQAGuidance(nn.Module):
    """Image Quality Assessment guidance"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = (x + 1) / 2
        brightness = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        brightness_var = torch.var(brightness, dim=[1, 2])
        brightness_loss = torch.mean((brightness_var - 0.1).abs())
        max_rgb = torch.max(x, dim=1)[0]
        min_rgb = torch.min(x, dim=1)[0]
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
        saturation_loss = -torch.mean(saturation)
        contrast = torch.std(x, dim=[2, 3])
        contrast_loss = -torch.mean(contrast)
        return brightness_loss + 0.5 * saturation_loss + 0.3 * contrast_loss


class PerceptualLoss(nn.Module):
    """Lightweight VGG19 perceptual loss optimized for PSNR"""
    def __init__(self):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

            self.slice1 = nn.Sequential(*list(vgg)[:4])
            self.slice2 = nn.Sequential(*list(vgg)[4:9])
            self.slice3 = nn.Sequential(*list(vgg)[9:18])

            for param in self.parameters():
                param.requires_grad = False

            logger.info("✓ VGG19 perceptual loss initialized (3 layers, optimized)")
        except Exception as e:
            logger.error(f"Failed to initialize VGG19: {e}")
            raise

    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = (y - mean) / std

        x1 = self.slice1(x)
        y1 = self.slice1(y)

        x2 = self.slice2(x1)
        y2 = self.slice2(y1)

        x3 = self.slice3(x2)
        y3 = self.slice3(y2)

        loss = (
            0.8 * F.l1_loss(x1, y1) +
            1.0 * F.l1_loss(x2, y2) +
            1.2 * F.l1_loss(x3, y3)
        ) / 3.0

        return loss


class GaussianDiffusion(nn.Module):
    def __init__(
        self, denoise_fn, image_size, channels=3, loss_type='l1', conditional=True,
        schedule_opt=None, learn_prompt_path=None, use_wavelet=False, use_cross_attention=True,
        pyramid_levels=3, adaptive_lambda=False, physical_guidance=True, edge_guidance=True,
        iqa_guidance=True, use_perceptual=True
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.image_size = image_size
        self.channels = channels
        self.conditional = conditional
        self.loss_type = loss_type
        self.use_wavelet = use_wavelet
        self.use_cross_attention = use_cross_attention
        self.pyramid_levels = pyramid_levels

        self.L_clip = None
        self.text_features = None
        self.physical_loss = None
        self.edge_loss = None
        self.iqa_loss = None
        self.lambda_net = None
        self.perceptual_loss = None

        if learn_prompt_path:
            try:
                from .clip_score import L_clip_from_feature, load_learned_prompt
                self.L_clip = L_clip_from_feature()
                self.text_features = load_learned_prompt(path=learn_prompt_path)
                logger.info("✓ CLIP guidance loaded")
            except Exception as e:
                logger.warning(f"⚠ Failed to load CLIP: {e}")

        self.physical_guidance_enabled = physical_guidance
        self.edge_guidance_enabled = edge_guidance
        self.iqa_guidance_enabled = iqa_guidance
        self.use_perceptual = use_perceptual

        if physical_guidance:
            self.physical_loss = PhysicalGuidance()
            logger.info("✓ Physical Guidance initialized")

        if edge_guidance:
            self.edge_loss = EdgeGuidance()
            logger.info("✓ Edge Guidance initialized")

        if iqa_guidance:
            self.iqa_loss = IQAGuidance()
            logger.info("✓ IQA Guidance initialized")

        if use_perceptual:
            try:
                self.perceptual_loss = PerceptualLoss()
                logger.info("✓ Perceptual loss enabled - PSNR boost active!")
            except Exception as e:
                logger.warning(f"⚠ Failed to load perceptual loss: {e}")
                self.use_perceptual = False

        if adaptive_lambda:
            try:
                from .adaptive_lambda import AdaptiveLambda
                self.lambda_net = AdaptiveLambda()
                logger.info("✓ Adaptive Lambda loaded")
            except Exception as e:
                logger.warning(f"⚠ Failed to load AdaptiveLambda: {e}")

        self.schedule_opt = schedule_opt
        self.visuals = {}
        self.data = {}
        self.current_step = 0
        self.loss_func = None

    def set_device(self, device):
        self.device = device
        self.set_loss(device)

        if self.schedule_opt:
            self.set_new_noise_schedule(self.schedule_opt, device)

        if self.physical_loss is not None:
            self.physical_loss = self.physical_loss.to(device)

        if self.edge_loss is not None:
            self.edge_loss = self.edge_loss.to(device)

        if self.iqa_loss is not None:
            self.iqa_loss = self.iqa_loss.to(device)

        if self.perceptual_loss is not None:
            self.perceptual_loss = self.perceptual_loss.to(device)
            logger.info("✓ Perceptual loss → GPU")

        if self.lambda_net is not None:
            self.lambda_net = self.lambda_net.to(device)

        if self.L_clip is not None:
            self.L_clip = self.L_clip.to(device)

        if self.text_features is not None:
            self.text_features = self.text_features.to(device)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        logger.info(f"✓ Loss function: {self.loss_type.upper()}")

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**schedule_opt)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = len(betas)

        for k, v in [
            ('betas', betas), ('alphas_cumprod', alphas_cumprod), ('alphas_cumprod_prev', alphas_cumprod_prev),
            ('sqrt_alphas_cumprod', np.sqrt(alphas_cumprod)), ('sqrt_one_minus_alphas_cumprod', np.sqrt(1. - alphas_cumprod)),
            ('sqrt_recip_alphas_cumprod', np.sqrt(1. / alphas_cumprod)), ('sqrt_recipm1_alphas_cumprod', np.sqrt(1. / alphas_cumprod - 1)),
            ('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)),
            ('posterior_log_variance_clipped', np.log(np.maximum(betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod), 1e-20))),
            ('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)),
            ('posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)),
            ('sqrt_alphas_cumprod_prev', np.sqrt(np.append(1., alphas_cumprod)))
        ]:
            self.register_buffer(k, to_torch(v))

        logger.info(f"✓ Noise schedule: {schedule_opt.get('schedule', 'unknown')} | steps: {self.num_timesteps}")

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        return continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise

    def forward(self, x_in, noise=None, global_step=0, total_steps=1e5):
        self.current_step = global_step
        x_start = x_in['HR']
        x_cond = x_in.get('LR', x_in.get('SR'))
        b, c, h, w = x_start.shape
        device = x_start.device

        if b == 0:
            return torch.tensor(0.0, device=device), {k: 0.0 for k in ['l_pix', 'l_percep', 'l_clip', 'l_phy', 'l_edge', 'l_iqa', 'l_ms', 'total', 'clip_w', 'lambda']}

        t_idx = int(np.random.randint(0, self.num_timesteps))
        low = self.sqrt_alphas_cumprod_prev[t_idx]
        high = self.sqrt_alphas_cumprod_prev[t_idx + 1] if t_idx < self.num_timesteps - 1 else self.sqrt_alphas_cumprod_prev[-1]
        cont = torch.rand(b, device=device) * (high - low) + low
        cont = cont.view(b, 1, 1, 1)

        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start, cont, noise)

        if x_cond.shape[2:] != x_noisy.shape[2:]:
            x_cond = F.interpolate(x_cond, size=(h, w), mode='bicubic', align_corners=False)

        x_input = torch.cat([x_cond, x_noisy], dim=1)
        t = torch.full((b,), t_idx, device=device, dtype=torch.long)
        pred_noise = self.denoise_fn(x_input, t)

        l_pix = 1.667 * 2000 * self.loss_func(pred_noise, noise) / (b * c * h * w)

        x_0 = (x_noisy - (1 - cont**2).sqrt() * pred_noise) / cont
        x_0_clamped = torch.clamp(x_0, -1, 1)

        # === PERCEPTUAL LOSS ===
        l_perceptual = torch.tensor(0.0, device=device)
        if self.perceptual_loss is not None and self.use_perceptual:
            try:
                l_perceptual = self.perceptual_loss(x_0_clamped, x_start)
            except Exception as e:
                logger.warning(f"Perceptual loss failed: {e}")

        step = float(global_step) / float(total_steps)
        clip_w = 16 * 20 * (0.1 + 0.9 * step)

        x_1 = (x_0_clamped + 1) / 2

        # === CLIP LOSS (FIXED) ===
        l_clip = torch.tensor(0.0, device=device)
        if self.L_clip is not None and self.text_features is not None:
            try:
                x_1_clip = F.interpolate(x_1, size=(224, 224), mode='bilinear', align_corners=False)
                l_clip = clip_w * self.L_clip(x_1_clip.float(), self.text_features.float())
            except Exception as e:
                logger.warning(f"CLIP loss failed: {e}")

        l_phy = torch.tensor(0.0, device=device)
        if self.physical_loss is not None:
            try:
                l_phy = self.physical_loss(x_0_clamped, x_cond)
            except Exception as e:
                logger.warning(f"Physical loss failed: {e}")

        l_edge = torch.tensor(0.0, device=device)
        if self.edge_loss is not None:
            try:
                l_edge = self.edge_loss(x_0_clamped)
            except Exception as e:
                logger.warning(f"Edge loss failed: {e}")

        l_iqa = torch.tensor(0.0, device=device)
        if self.iqa_loss is not None:
            try:
                l_iqa = self.iqa_loss(x_0_clamped)
            except Exception as e:
                logger.warning(f"IQA loss failed: {e}")

        l_ms = torch.tensor(0.0, device=device)
        if h % 2 == 0 and w % 2 == 0 and h >= 64:
            try:
                x_noisy_p1 = F.avg_pool2d(x_noisy, 2)
                x_cond_p1 = F.avg_pool2d(x_cond, 2)
                x_input_p1 = torch.cat([x_cond_p1, x_noisy_p1], 1)
                t_p1 = torch.full((b,), max(0, t_idx // 2), device=device, dtype=torch.long)
                pred_p1 = self.denoise_fn(x_input_p1, t_p1)
                x_recon_p1 = F.interpolate(pred_p1, scale_factor=2, mode='bilinear', align_corners=False)
                l_ms = 0.3 * F.mse_loss(x_recon_p1, x_start)
            except Exception as e:
                logger.warning(f"Multi-scale loss failed: {e}")

        lambda_t = 0.4
        if self.lambda_net:
            try:
                lambda_t = float(self.lambda_net(x_noisy, x_cond).mean().item())
                lambda_t = max(0.1, min(0.8, lambda_t))
            except Exception as e:
                logger.warning(f"Lambda computation failed: {e}")

        def to_float(x):
            if isinstance(x, torch.Tensor):
                try:
                    return float(x.item())
                except:
                    return float(x.detach().cpu().mean().item())
            return float(x)

        # === TOTAL LOSS (KEEP AS TENSOR!) ===
        total = (
            0.78 * l_pix +
            0.04 * l_perceptual +
            0.12 * l_ms +
            0.02 * lambda_t * l_clip +
            0.01 * l_phy +
            0.01 * l_edge +
            0.02 * l_iqa
        )

        logs = {
            'l_pix': to_float(l_pix),
            'l_percep': to_float(l_perceptual),
            'l_clip': to_float(l_clip),
            'l_phy': to_float(l_phy),
            'l_edge': to_float(l_edge),
            'l_iqa': to_float(l_iqa),
            'l_ms': to_float(l_ms),
            'total': to_float(total),
            'clip_w': float(clip_w),
            'lambda': float(lambda_t)
        }

        return total, logs

    @torch.no_grad()
    def super_resolution(self, x_in, sample_num=5):
        self.denoise_fn.eval()

        x_lr = x_in.get('LR', x_in.get('SR'))
        b, c, h_lr, w_lr = x_lr.shape
        device = x_lr.device

        h, w = h_lr * 8, w_lr * 8
        x_cond = F.interpolate(x_lr, size=(h, w), mode='bicubic', align_corners=False)

        x_t = torch.randn(b, c, h, w, device=device)

        timesteps = torch.linspace(self.num_timesteps - 1, 0, sample_num, dtype=torch.long, device=device)

        for i, t_curr in enumerate(timesteps):
            t_batch = torch.full((b,), int(t_curr.item()), device=device, dtype=torch.long)
            x_input = torch.cat([x_cond, x_t], dim=1)

            pred_noise = self.denoise_fn(x_input, t_batch)

            alpha_t = self.alphas_cumprod[int(t_curr.item())]

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_t_next = self.alphas_cumprod[int(t_next.item())]

                x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                x_0_pred = torch.clamp(x_0_pred, -1, 1)

                x_t = torch.sqrt(alpha_t_next) * x_0_pred + torch.sqrt(1 - alpha_t_next) * pred_noise
            else:
                x_t = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                x_t = torch.clamp(x_t, -1, 1)

        return x_t

    def p_sample(self, x, t):
        t_batch = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
        pred_noise = self.denoise_fn(x, t_batch)

        alpha_t = self.alphas_cumprod[int(t)]
        alpha_t_prev = self.alphas_cumprod_prev[int(t)]

        x_0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)

        mean = self.posterior_mean_coef1[int(t)] * x_0_pred + self.posterior_mean_coef2[int(t)] * x

        if int(t) > 0:
            noise = torch.randn_like(x)
            variance = self.posterior_variance[int(t)]
            return mean + torch.sqrt(variance) * noise
        else:
            return mean