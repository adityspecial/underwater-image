import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import logging

logger = logging.getLogger('base')

# ========================================
# 1. IMAGE CONVERSION (tensor → numpy)
# ========================================
def tensor2img(tensor, min_max=(-1, 1)):
    """
    Convert [-1,1] tensor (C,H,W) or (B,C,H,W) to uint8 numpy (H,W,C)
    """
    if tensor is None:
        return None
    tensor = tensor.clamp(*min_max).cpu()
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # [0,1]
    img = tensor.numpy()
    if img.ndim == 4:
        img = img[0]  # take first image
    img = np.transpose(img, (1, 2, 0))  # HWC
    img = (img * 255).round().astype(np.uint8)
    return img

# ========================================
# 2. SAFE RESIZE (for UIQM/UCIQE)
# ========================================
def safe_resize(img, min_size=64):
    if img is None or min(img.shape[:2]) >= min_size:
        return img
    h, w = img.shape[:2]
    scale = min_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# ========================================
# 3. PSNR
# ========================================
def calculate_psnr(img1, img2, data_range=255):
    """
    img1, img2: uint8 numpy (H,W,C) or (H,W)
    """
    if img1 is None or img2 is None:
        return 0.0
    try:
        return peak_signal_noise_ratio(img1, img2, data_range=data_range)
    except Exception as e:
        logger.warning(f"PSNR failed: {e}")
        return 0.0

# ========================================
# 4. SSIM
# ========================================
def calculate_ssim(img1, img2, data_range=255):
    """
    Safe SSIM with dynamic window size
    """
    if img1 is None or img2 is None:
        return 0.0
    h, w = img1.shape[:2]
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return 0.0
    try:
        return structural_similarity(
            img1, img2,
            win_size=win_size,
            multichannel=(img1.ndim == 3),
            data_range=data_range
        )
    except Exception as e:
        logger.warning(f"SSIM failed: {e}")
        return 0.0

# ========================================
# 5. UIQM (Underwater Image Quality Measure)
# ========================================
def calculate_uiqm(img):
    """
    UIQM = 0.0282×UICM + 0.2953×UISM + 3.5753×UIConM
    """
    if img is None:
        return 0.0
    try:
        img = safe_resize(img)
        img = img.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        R, G, B = img[..., 0], img[..., 1], img[..., 2]

        # UICM: colorfulness
        rg = R - G
        yb = 0.5 * (R + G) - B
        uicm = abs(np.mean(rg)) + abs(np.mean(yb))

        # UISM: sharpness
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        grad_x = np.gradient(gray, axis=0)
        grad_y = np.gradient(gray, axis=1)
        uism = np.mean(np.sqrt(grad_x**2 + grad_y**2))

        # UIConM: contrast
        uiconm = np.log2(1 + np.std(img)) if np.std(img) > 0 else 0

        return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    except Exception as e:
        logger.warning(f"UIQM failed: {e}")
        return 0.0

# ========================================
# 6. UCIQE (Underwater Color Image Quality Evaluation)
# ========================================
def calculate_uciqe(img):
    """
    UCIQE = 0.4680×σ_c + 0.2745×μ_s + 0.2576×μ_l
    """
    if img is None:
        return 0.0
    try:
        img = safe_resize(img)
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

        sigma_c = np.std(A) + np.std(B)
        mu_s = np.mean(np.abs(A - 128) + np.abs(B - 128))
        L_mean = np.mean(L)

        return 0.4680 * sigma_c + 0.2745 * mu_s + 0.2576 * L_mean
    except Exception as e:
        logger.warning(f"UCIQE failed: {e}")
        return 0.0

# ========================================
# 7. NO save_img() → NO cv2 ERROR
# ========================================
# REMOVED: save_img() → prevents OpenCV crash
