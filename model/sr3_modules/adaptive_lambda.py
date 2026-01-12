# @title model/sr3_modules/adaptive_lambda.py
import torch.nn as nn

class AdaptiveLambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
    def forward(self, x_t, cond):
        return self.net(torch.cat([x_t, cond], dim=1))