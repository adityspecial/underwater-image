import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch

class LRHRDataset(data.Dataset):
    def __init__(self, dataroot_HR=None, dataroot_LR=None, dataroot=None,
                 l_resolution=64, r_resolution=256, split='train', data_len=-1, need_LR=False):
        self.data_len = data_len
        self.split = split
        self.need_LR = need_LR
        self.l_resolution = l_resolution
        self.r_resolution = r_resolution

        # === SUPPORT BOTH ===
        if dataroot_HR and dataroot_LR:
            self.hr_path = dataroot_HR
            self.lr_path = dataroot_LR
        elif dataroot:
            self.hr_path = os.path.join(dataroot, 'HR')
            self.lr_path = os.path.join(dataroot, 'LR')
        else:
            raise ValueError("Must provide dataroot_HR/dataroot_LR or dataroot")

        self.hr_list = sorted([os.path.join(self.hr_path, f) for f in os.listdir(self.hr_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_list = sorted([os.path.join(self.lr_path, f) for f in os.listdir(self.lr_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if data_len > 0:
            self.data_len = min(data_len, len(self.hr_list))
        else:
            self.data_len = len(self.hr_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Get the file paths
        hr_path = self.hr_list[index % len(self.hr_list)]
        lr_path = self.lr_list[index % len(self.lr_list)]
        
        img_HR = Image.open(hr_path).convert("RGB")
        img_LR = Image.open(lr_path).convert("RGB") if self.need_LR else None

        # Resize
        if img_HR.size != (self.r_resolution, self.r_resolution):
            img_HR = img_HR.resize((self.r_resolution, self.r_resolution), Image.BICUBIC)
        if img_LR is not None and img_LR.size != (self.l_resolution, self.l_resolution):
            img_LR = img_LR.resize((self.l_resolution, self.l_resolution), Image.BICUBIC)

        img_HR = (np.array(img_HR).astype(np.float32) / 127.5) - 1.0
        img_LR = (np.array(img_LR).astype(np.float32) / 127.5) - 1.0 if img_LR is not None else None

        img_HR = torch.from_numpy(np.transpose(img_HR, (2, 0, 1)))
        img_LR = torch.from_numpy(np.transpose(img_LR, (2, 0, 1))) if img_LR is not None else None

        # Return dictionary with paths included
        return {
            'HR': img_HR, 
            'LR': img_LR,
            'HR_path': hr_path,
            'LR_path': lr_path
        }