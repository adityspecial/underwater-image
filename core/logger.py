# @title core/logger.py — FINAL: NO dict_to_nonedict
import argparse
import logging
import yaml
import os
import sys
from datetime import datetime

def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '--debug', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Setup logger
    log_dir = opt['path']['log']
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('base')
    logger.info(f'Log file: {log_file}')

    return opt