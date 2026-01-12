# model/networks.py — FIXED: Proper CLIP path detection
import logging
import torch
import torch.nn as nn

logger = logging.getLogger('base')


def define_G(opt):
    """
    Creates GaussianDiffusion with the correct backbone.
    Your config must have: "which_model_G": "mamba_unet"
    """
    from .sr3_modules.diffusion import GaussianDiffusion
    
    model_type = opt['model'].get('which_model_G', 'sr3').lower()
    model_config = opt['model']
    unet_config = model_config.get('unet', {})

    # ============================================================
    # STEP 1: Create the denoising backbone
    # ============================================================
    
    if model_type == 'sr3':
        from .hybrid_unet import HybridUNet
        denoise_fn = HybridUNet(
            in_channel=unet_config.get('in_channel', 6),
            out_channel=unet_config.get('out_channel', 3),
            inner_channel=unet_config.get('inner_channel', 64),
            channel_mults=unet_config.get('channel_mults', [1, 2, 4, 8]),
            res_blocks=unet_config.get('res_blocks', 3),
            dropout=unet_config.get('dropout', 0.05),
            num_heads=unet_config.get('num_heads', 8),
            window_size=unet_config.get('window_size', 8),
        )
        logger.info('✓ Backbone: HybridUNet (Swin Transformer)')

    elif model_type == 'mamba_unet':
        # Try both possible class names
        try:
            from .mamba_hybrid_unet import MambaEnhancedUNet
            UNetClass = MambaEnhancedUNet
            use_mamba_enhanced = True
        except ImportError:
            from .mamba_unet import MambaUNet
            UNetClass = MambaUNet
            use_mamba_enhanced = False
            logger.info('Using MambaUNet from mamba_unet.py')
        
        if use_mamba_enhanced:
            # MambaEnhancedUNet uses singular parameter names
            denoise_fn = UNetClass(
                in_channel=unet_config.get('in_channel', 6),
                out_channel=unet_config.get('out_channel', 3),
                inner_channel=unet_config.get('inner_channel', 64),
                channel_mults=unet_config.get('channel_mults', [1, 2, 4, 8]),
                res_blocks=unet_config.get('res_blocks', 2),
                dropout=unet_config.get('dropout', 0.0),
                with_time_emb=True,
                use_mamba=unet_config.get('use_mamba', True),
            )
        else:
            # MambaUNet uses plural parameter names
            denoise_fn = UNetClass(
                in_channels=6,
                out_channels=3,
                depths=[2, 4, 6, 4],
                dims=[96, 192, 384, 768],
                d_state=16
            )
        
        # Move to GPU before wrapping
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        denoise_fn = denoise_fn.to(device)
        
        logger.info('✓ Backbone: MambaEnhancedUNet (State Space Model)')
        logger.info(f'✓ Device: {device}')

    else:
        raise ValueError(f"Unknown which_model_G: {model_type}. Use 'sr3' or 'mamba_unet'")

    # ============================================================
    # STEP 2: Get diffusion config (flexible structure)
    # ============================================================
    
    # Support both old and new config structures
    if 'diffusion' in model_config:
        diffusion_config = model_config['diffusion']
    else:
        diffusion_config = {}
    
    # Get beta_schedule (can be at model level or diffusion level)
    beta_schedule = model_config.get('beta_schedule')
    if beta_schedule is None and 'beta_schedule' in diffusion_config:
        beta_schedule = diffusion_config['beta_schedule']
    
    # CRITICAL FIX: Check MULTIPLE locations for learn_prompt_path
    learn_prompt_path = None
    
    # Priority 1: Inside diffusion config
    if 'learn_prompt_path' in diffusion_config:
        learn_prompt_path = diffusion_config['learn_prompt_path']
        logger.info(f"🔍 CLIP path found in diffusion: {learn_prompt_path}")
    
    # Priority 2: Inside prompt config
    elif 'prompt' in model_config and 'learn_prompt_path' in model_config['prompt']:
        learn_prompt_path = model_config['prompt']['learn_prompt_path']
        logger.info(f"🔍 CLIP path found in prompt: {learn_prompt_path}")
    
    # Priority 3: Direct in model config
    elif 'learn_prompt_path' in model_config:
        learn_prompt_path = model_config['learn_prompt_path']
        logger.info(f"🔍 CLIP path found in model: {learn_prompt_path}")
    
    else:
        logger.warning("⚠️ No CLIP prompt path found - CLIP guidance disabled")
    
    # Check if file exists
    if learn_prompt_path:
        import os
        if os.path.exists(learn_prompt_path):
            logger.info(f"✓ CLIP file exists: {learn_prompt_path}")
        else:
            logger.error(f"❌ CLIP file NOT FOUND: {learn_prompt_path}")
            learn_prompt_path = None
    
    # Get guidance settings (can be at model or diffusion level)
    def get_guidance_setting(key, default=False):
        # Try diffusion level first, then model level
        if key in diffusion_config:
            return diffusion_config.get(key, default)
        return model_config.get(key, default)
    
    # ============================================================
    # STEP 3: Wrap in Gaussian Diffusion
    # ============================================================
    
    netG = GaussianDiffusion(
        denoise_fn=denoise_fn,
        image_size=diffusion_config.get('image_size', 256),
        channels=diffusion_config.get('channels', 3),
        loss_type=diffusion_config.get('loss_type', 'l1'),
        conditional=diffusion_config.get('conditional', True),
        schedule_opt=beta_schedule.get('train') if beta_schedule else None,
        
        # Guidance options (flexible lookup)
        learn_prompt_path=learn_prompt_path,
        use_wavelet=get_guidance_setting('use_wavelet', False),
        use_cross_attention=get_guidance_setting('use_cross_attention', False),
        pyramid_levels=get_guidance_setting('pyramid_levels', 3),
        adaptive_lambda=get_guidance_setting('adaptive_lambda', False),
        
        # Loss components (CRITICAL for PSNR)
        physical_guidance=get_guidance_setting('physical_guidance', True),
        edge_guidance=get_guidance_setting('edge_guidance', True),
        iqa_guidance=get_guidance_setting('iqa_guidance', True),
        use_perceptual=get_guidance_setting('use_perceptual', True),
    )
    
    logger.info('✓ Diffusion wrapper created')
    logger.info(f'  - Image size: {diffusion_config.get("image_size", 256)}')
    logger.info(f'  - Loss type: {diffusion_config.get("loss_type", "l1")}')
    logger.info(f'  - Perceptual loss: {get_guidance_setting("use_perceptual", True)}')
    logger.info(f'  - Physical guidance: {get_guidance_setting("physical_guidance", True)}')
    logger.info(f'  - Edge guidance: {get_guidance_setting("edge_guidance", True)}')
    logger.info(f'  - IQA guidance: {get_guidance_setting("iqa_guidance", True)}')
    logger.info(f'  - CLIP prompt: {"✅ YES" if learn_prompt_path else "❌ NO"}')
    
    # ============================================================
    # STEP 4: Data Parallel (if multi-GPU)
    # ============================================================
    
    gpu_ids = opt.get('gpu_ids', [])
    if gpu_ids and len(gpu_ids) > 1:
        assert torch.cuda.is_available(), "CUDA not available for multi-GPU"
        netG = nn.DataParallel(netG, device_ids=gpu_ids)
        logger.info(f'✓ DataParallel enabled on GPUs: {gpu_ids}')
    
    return netG


def print_network(net):
    """
    Print network architecture and parameter count
    """
    if isinstance(net, nn.DataParallel):
        net = net.module
    
    num_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    logger.info(f'Network: {net.__class__.__name__}')
    logger.info(f'Total parameters: {num_params / 1e6:.2f}M')
    logger.info(f'Trainable parameters: {trainable_params / 1e6:.2f}M')
    
    return num_params