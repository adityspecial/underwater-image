import os
import torch
import torch.nn as nn
import logging

logger = logging.getLogger('base')

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] and len(opt['gpu_ids']) > 0 else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0
        self.netG = None

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save_network(self, step):
        save_path = f"{self.opt['path']['checkpoint']}/step_{step}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.netG.state_dict(), save_path)
        logger.info(f"Checkpoint saved: {save_path}")

    def load_network(self, load_path):
        if os.path.exists(load_path):
            state_dict = torch.load(load_path, map_location=self.device)
            self.netG.load_state_dict(state_dict)
            logger.info(f"Model loaded from {load_path}")
        else:
            logger.warning(f"Checkpoint not found: {load_path}")

    def set_device(self, x):
        """
        Move tensors to device, but skip non-tensor items like paths (strings).
        """
        if isinstance(x, dict):
            for key, item in x.items():
                # Skip string paths (HR_path, LR_path, etc.)
                if isinstance(item, str):
                    continue
                # Skip lists of strings
                if isinstance(item, list) and len(item) > 0 and isinstance(item[0], str):
                    continue
                # Move tensors to device
                if item is not None and torch.is_tensor(item):
                    x[key] = item.to(self.device)
                # Handle None values
                elif item is None:
                    x[key] = None
        elif isinstance(x, list):
            # Only move tensors in the list
            x = [item.to(self.device) if item is not None and torch.is_tensor(item) else item for item in x]
        else:
            # Move single tensor
            x = x.to(self.device) if x is not None and torch.is_tensor(x) else x
        return x

    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n