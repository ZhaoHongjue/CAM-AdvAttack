import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from scipy.optimize import differential_evolution as diffevo

from .base import BaseAttack

class OnePixel(BaseAttack):
    '''
    Paper: One Pixel Attack for Fooling Deep Neural Networks
    '''
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)  
        
    def add_perturb(
        self, 
        onepix_perturb: np.ndarray,
        img_tensor: torch.Tensor
    ) -> torch.Tensor:
        img_clone = img_tensor.clone().to(self.device)
        if img_clone.dim() == 3:
            img_clone = img_clone.unsqueeze(0)
            
        onepix_perturb_tensor = torch.tensor(onepix_perturb, dtype = torch.float32)
        coord = torch.as_tensor(onepix_perturb_tensor[:2], dtype = int)
        perturb = onepix_perturb_tensor[2:]
        img_clone[:, :, coord[0], coord[1]] += perturb.to(self.device)
        return img_clone
    
    def predict(
        self, 
        onepix_perturb: np.ndarray,
        img_tensor: torch.Tensor,
        label: int,
    ):
        img_clone = self.add_perturb(onepix_perturb, img_tensor)
        img_clone = img_clone.to(self.device)
        with torch.no_grad():
            prob = F.softmax(
                self.model(img_clone), dim = 1
            ).cpu().numpy()[:, label]
            return prob
    
    def __call__(
        self, 
        img_tensor: torch.Tensor,
        label: int, 
        maxiter: int = 100, 
        popsize: int = 400
    ) -> torch.Tensor:
        bounds = [
            (0, img_tensor.shape[-2]), 
            (0, img_tensor.shape[-1]), 
            (0, 1), (0, 1), (0, 1),
        ]
        popsize = max(1, popsize)
        opt_fn = lambda x: self.predict(x, img_tensor, label)
        diffevo_ret = diffevo(
            opt_fn, bounds, maxiter = maxiter, 
            popsize = popsize, init = 'random'
        )
        # print(diffevo_ret)
        # print(diffevo_ret.x)
        onepix_perturb = diffevo_ret.x
        return self.add_perturb(onepix_perturb, img_tensor).squeeze(0).cpu()
        