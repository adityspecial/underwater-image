# train.py — AUTO-RESUME + Proper training (full version)
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from tensorboardX import SummaryWriter
import os
import glob
import re
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

logger = logging.getLogger('base')


# -------------------------
# AUTO RESUME HELPERS
# -------------------------
def find_latest_checkpoint(ckpt_dir):
    """
    Return the newest generator checkpoint path (gen_ckpt) or None.
    Supports multiple formats.
    """
    if not os.path.exists(ckpt_dir):
        return None

    patterns = [
        "I*_E*_gen.pth",
        "iter_*_gen.pth",
        "iter_*.pth",
        "I*.pth"
    ]

    candidates = []
    for p in patterns:
        candidates += glob.glob(os.path.join(ckpt_dir, p))

    if len(candidates) == 0:
        return None

    def extract_iter(path):
        name = os.path.basename(path)
        if name.startswith("iter_"):
            m = re.search(r"iter_(\d+)", name)
            if m:
                return int(m.group(1))
        if name.startswith("I"):
            m = re.match(r"I(\d+)", name)
            if m:
                return int(m.group(1))
        m = re.search(r"(\d+)", name)
        if m:
            return int(m.group(1))
        return 0

    gen_candidates = [c for c in candidates if "_gen.pth" in os.path.basename(c)]
    if len(gen_candidates) > 0:
        gen_candidates.sort(key=lambda x: extract_iter(x))
        return gen_candidates[-1]

    iter_candidates = [c for c in candidates if os.path.basename(c).startswith("iter_")]
    if len(iter_candidates) > 0:
        iter_candidates.sort(key=lambda x: extract_iter(x))
        return iter_candidates[-1]

    candidates.sort(key=lambda x: extract_iter(x))
    return candidates[-1]


# ============================================================
# METRICS
# ============================================================
def safe_resize(img, min_size=64):
    if img is None:
        return None
    h, w = img.shape[:2]
    if min(h, w) < min_size:
        scale = min_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img


def calculate_uiqm(img):
    if img is None:
        return 0.0

    img = safe_resize(img)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    uicm = abs(np.mean(R - G)) + abs(np.mean(0.5 * (R + G) - B))

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    grad_x = np.gradient(gray, axis=0)
    grad_y = np.gradient(gray, axis=1)
    uism = np.mean(np.sqrt(grad_x**2 + grad_y**2))

    uiconm = np.log2(1 + np.std(img)) if np.std(img) > 0 else 0.0

    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def calculate_uciqe(img):
    if img is None:
        return 0.0

    img = safe_resize(img)
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

    if img_uint8.size == 0 or np.all(img_uint8 == 0):
        return 0.0

    try:
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    except:
        return 0.0

    L = lab[:, :, 0].astype(np.float64)
    A = lab[:, :, 1].astype(np.float64)
    B = lab[:, :, 2].astype(np.float64)

    sigma_c = float(np.clip(np.std(A) + np.std(B), 0, 1000))
    contrast_L = float(np.clip(np.max(L) - np.min(L), 0, 255))
    mu_s = float(np.clip(np.mean(np.abs(A - 128)) + np.mean(np.abs(B - 128)), 0, 500))

    uciqe = 0.4680 * sigma_c + 0.2745 * contrast_L + 0.2576 * mu_s
    if not np.isfinite(uciqe):
        return 0.0

    return float(np.clip(uciqe, 0, 150))


def safe_ssim(img1, img2):
    if img1 is None or img2 is None:
        return 0.0

    img1 = safe_resize(img1)
    img2 = safe_resize(img2)

    h, w = img1.shape[:2]
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return 0.0

    try:
        return float(structural_similarity(img1, img2, win_size=win_size, channel_axis=-1, data_range=255))
    except:
        return 0.0


def calculate_psnr(img1, img2):
    try:
        return float(peak_signal_noise_ratio(img1, img2, data_range=255))
    except:
        return 0.0


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/pretrain_uieb_v4.json')
    args = parser.parse_args()

    opt = Logger.parse(args)

    # === DATASETS ===
    train_set = None
    val_set = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)

    logger.info(f"Train: {len(train_set):,d} images | Val: {len(val_set):,d} images")

    # ============================================================
    # AUTO-RESUME
    # ============================================================
    ckpt_dir = opt['path']['checkpoint']
    latest_gen = find_latest_checkpoint(ckpt_dir)

    if latest_gen:
        logger.info(f"🔄 Found checkpoint: {latest_gen}")
        opt['path']['resume_state'] = latest_gen
    else:
        opt['path']['resume_state'] = None
        logger.info("🚀 No checkpoint found — starting fresh.")

    diffusion = Model.create_model(opt)
    current_step = getattr(diffusion, 'current_step', 0)
    n_iter = opt['train']['n_iter']

    writer = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # ============================================================
    # TRAIN LOOP
    # ============================================================
    for epoch in range(200000):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break

            diffusion.feed_data(train_data)
            diffusion.current_step = current_step
            diffusion.n_iter = n_iter
            diffusion.optimize_parameters()

            logs = diffusion.get_current_log()

            # ---------- LOGGING ----------
            if current_step % opt['train']['print_freq'] == 0:
                log_s = f"[Epoch {epoch} | Iter {current_step}]  " + \
                        " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                logger.info(log_s)

            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", v, current_step)

            # ---------- SAVE CHECKPOINT ----------
            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                diffusion.save_network(epoch, current_step)

            # =====================================================
            # VALIDATION + SAVE SR IMAGES
            # =====================================================
            if current_step % opt['train']['val_freq'] == 0:
                logger.info(f"=== VALIDATION @ iter {current_step} ===")

                psnr_vals, ssim_vals, uiqm_vals, uciqe_vals = [], [], [], []

                diffusion.netG.eval()
                with torch.no_grad():
                    for val_idx, val_data in enumerate(val_loader):
                        # Debug: print available keys on first validation
                        if val_idx == 0:
                            logger.info(f"Val data keys: {list(val_data.keys())}")
                        diffusion.feed_data(val_data)
                        diffusion.test()

                        visuals = diffusion.get_current_visuals()
                        sr = visuals['SR'][0].clamp(-1, 1).cpu().numpy()
                        hr = visuals['HR'][0].cpu().numpy()

                        sr = np.transpose(sr, (1, 2, 0))
                        hr = np.transpose(hr, (1, 2, 0))

                        sr_uint8 = ((sr + 1) * 127.5).astype(np.uint8)
                        hr_uint8 = ((hr + 1) * 127.5).astype(np.uint8)

                        # -------------------------
                        # SAVE SR IMAGE
                        # -------------------------
                        save_root = opt['path']['results']
                        save_dir = os.path.join(save_root, f"iter_{current_step}")
                        os.makedirs(save_dir, exist_ok=True)

                        # Handle different possible path keys
                        if 'HR_path' in val_data and val_data['HR_path']:
                            img_name = os.path.basename(val_data['HR_path'][0])
                        elif 'LR_path' in val_data and val_data['LR_path']:
                            img_name = os.path.basename(val_data['LR_path'][0])
                        elif 'path' in val_data and val_data['path']:
                            img_name = os.path.basename(val_data['path'][0])
                        else:
                            # Fallback: generate filename from index
                            img_name = f"val_{val_idx:04d}.png"

                        save_path = os.path.join(save_dir, img_name)

                        cv2.imwrite(save_path, cv2.cvtColor(sr_uint8, cv2.COLOR_RGB2BGR))
                        logger.info(f"Saved SR: {save_path}")

                        # -------------------------
                        # METRICS
                        # -------------------------
                        psnr_vals.append(calculate_psnr(hr_uint8, sr_uint8))
                        ssim_vals.append(safe_ssim(hr_uint8, sr_uint8))
                        uiqm_vals.append(calculate_uiqm(sr_uint8))
                        uciqe_vals.append(calculate_uciqe(sr_uint8))

                avg_psnr = np.mean(psnr_vals)
                avg_ssim = np.mean(ssim_vals)
                avg_uiqm = np.mean(uiqm_vals)
                avg_uciqe = np.mean(uciqe_vals)

                logger.info(
                    f"VAL | UIQM: {avg_uiqm:.3f} | UCIQE: {avg_uciqe:.3f}  "
                    f"| PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f}"
                )

                writer.add_scalar("val/UIQM", avg_uiqm, current_step)
                writer.add_scalar("val/UCIQE", avg_uciqe, current_step)
                writer.add_scalar("val/PSNR", avg_psnr, current_step)
                writer.add_scalar("val/SSIM", avg_ssim, current_step)

                diffusion.netG.train()

        if current_step > n_iter:
            logger.info("🎉 TRAINING COMPLETE!")
            break

    writer.close()