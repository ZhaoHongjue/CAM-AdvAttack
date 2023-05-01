import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

class BaseAttack:
    def __init__(
        self,
        model: nn.Module,
        use_cuda: bool = True
    ) -> None:
        self.model = model
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        
    def __call__(
        self, 
        img_tensor: torch.Tensor, 
        eps: float
    ) -> torch.Tensor:
        raise NotImplementedError