import torch
from torch import nn

from torchvision import transforms

import numpy as np

def Clip(
    img_tensor: torch.Tensor, 
    img_attack: torch.Tensor,
    eps: float
) -> torch.Tensor:
    pass

def DiverseInput(img_tensor: torch.Tensor, prob: float) -> torch.Tensor:
    if not (0 <= prob <=1):
        raise ValueError('`prob` should be in [0, 1]!')
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(
            size = 224,
            scale = (0.1, 1)
        )
    ])
    if np.random.rand() < prob:
        return tfm(img_tensor)
    else:
        return img_tensor
    
