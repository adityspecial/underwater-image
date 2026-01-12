# @title FIXED diffusion.py — CLIP-UIE + UIEB + NOVELTY
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from . import clip_score

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

def exists(x): return x is not None
def default(val, d): return val if exists(val) else (d() if isfunction(d) else d)

# === PHYSICAL LOSS ===
def physical_loss(pred, lr_up):
    gray = lr_up.mean(1, keepdim=True)
    t = torch.exp(-0.1 * (1 - gray))
    A = lr_up.mean(dim=[2,3], keepdim=True)
    J = (pred - A) / (t + 1e-6) + A
    return F.l1_loss(pred, J)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        learn_prompt_path='/data/liusx/Pycharm/CLIP-LIT-main/train0/snapshots_prompt_train0/iter_100000.pth'
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.L_clip = clip_score.L_clip_from_feature()
        self.L_clip_MSE = clip_score.L_clip_MSE()
        self.text_features = clip_score.load_learned_prompt(path=learn_prompt_path)

        # === FIX: Only set schedule if opt is given AND device is ready ===
        if schedule_opt is not None:
            self.schedule_opt = schedule_opt  # Store for later

    def set_device(self, device):
        """Called by model.py after GPU is set"""
        self.device = device
        if hasattr(self, 'schedule_opt'):
            self.set_new_noise_schedule(self.schedule_opt, device)

    def set_loss(self, device):
        self.loss_func = nn.L1Loss(reduction='mean').to(device) if self.loss_type == 'l1' else nn.MSELoss(reduction='mean').to(device)

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**schedule_opt)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        for k, v in [
            ('betas', betas), ('alphas_cumprod', alphas_cumprod), ('alphas_cumprod_prev', alphas_cumprod_prev),
            ('sqrt_alphas_cumprod', np.sqrt(alphas_cumprod)),
            ('sqrt_one_minus_alphas_cumprod', np.sqrt(1. - alphas_cumprod)),
            ('log_one_minus_alphas_cumprod', np.log(1. - alphas_cumprod)),
            ('sqrt_recip_alphas_cumprod', np.sqrt(1. / alphas_cumprod)),
            ('sqrt_recipm1_alphas_cumprod', np.sqrt(1. / alphas_cumprod - 1)),
            ('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)),
            ('posterior_log_variance_clipped', np.log(np.maximum(betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod), 1e-20))),
            ('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)),
            ('posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        ]: self.register_buffer(k, to_torch(v))

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (continuous_sqrt_alpha_cumprod * x_start + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise)

    def q_predict_start_from_noise(self, x_t, continuous_sqrt_alpha_cumprod, noise):
        return ((1. / continuous_sqrt_alpha_cumprod) * x_t - ((1 - continuous_sqrt_alpha_cumprod**2).sqrt()) * (1. / continuous_sqrt_alpha_cumprod) * noise)

    def p_losses(self, x_in, noise=None, global_step=0, total_steps=1e5):
        x_start = x_in['HR']
        x_cond = x_in['SR']
        b, c, h, w = x_start.shape

        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(x_start.device).view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        x_input = torch.cat([x_cond, x_noisy], dim=1)
        x_recon = self.denoise_fn(x_input, continuous_sqrt_alpha_cumprod)

        # === PIXEL LOSS ===
        l_pix = 1.667 * 2000 * self.loss_func(noise, x_recon)

        # === CLIP LOSS (CURRICULUM) ===
        step = global_step / total_steps
        clip_weight = 16 * 20 * (0.1 + 0.9 * step)
        x_0 = self.q_predict_start_from_noise(x_noisy, continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), x_recon)
        x_1 = (torch.clamp(x_0, -1, 1) + 1) / 2
        loss_clip = clip_weight * self.L_clip(x_1, self.text_features)

        # === PHYSICAL LOSS ===
        loss_phy = physical_loss(x_0, x_cond)

        # === MULTI-SCALE LOSS ===
        x_noisy_p1 = F.avg_pool2d(x_noisy, 2)
        x_cond_p1 = F.avg_pool2d(x_cond, 2)
        noise_p1 = F.avg_pool2d(noise, 2)
        cont_p1 = continuous_sqrt_alpha_cumprod[:, ::2] if t % 2 == 0 else continuous_sqrt_alpha_cumprod[:, 1::2]
        x_input_p1 = torch.cat([x_cond_p1, x_noisy_p1], 1)
        x_recon_p1 = F.interpolate(self.denoise_fn(x_input_p1, cont_p1), scale_factor=2)
        loss_p1 = 0.3 * F.mse_loss(x_recon_p1, x_start)

        # === FINAL LOSS ===
        total_loss = 0.6 * l_pix + 0.4 * loss_clip + 0.1 * loss_phy + 0.1 * loss_p1

        logs = {
            'l_pix': l_pix.item(),
            'l_clip': loss_clip.item(),
            'l_phy': loss_phy.item(),
            'l_p1': loss_p1.item(),
            'total': total_loss.item(),
            'clip_w': clip_weight
        }

        return total_loss, logs

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)[0]