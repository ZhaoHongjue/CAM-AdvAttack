import torch
from torch import nn
import numpy as np
from typing import Tuple

from .base import BaseAttack


class JSMA(BaseAttack):
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        
    def __call__(
        self, 
        img_tensor: torch.Tensor, 
        target_label: int,
        theta: float = 0.05,
        max_iter: int = 100,
    ) -> torch.Tensor:
        i = 0
        search_domain = torch.lt(img_tensor, 1.00).reshape(-1).to(self.device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(img_tensor.to(self.device))
            curr_pred = torch.argmax(logits[0]).item()
        
        attack_img = img_tensor.clone()
        while (i < max_iter) and (curr_pred != target_label) and (search_domain.sum() != 0):
            jacob = self._get_jacob(img_tensor)
            p1, p2 = self.saliency_map(jacob, target_label, search_domain)
            attack_img_flatten = attack_img.reshape(-1)
            
            attack_img_flatten[p1] += theta
            attack_img_flatten[p2] += theta
            attack_img_flatten = torch.clamp(attack_img_flatten, 0.0, 1.0)
            
            if attack_img_flatten[p1] == 1.0 or attack_img_flatten[p1] == 0.0:
                search_domain[p1] = False
            if attack_img_flatten[p2] == 1.0 or attack_img_flatten[p2] == 0.0:
                search_domain[p2] = False
            
            attack_img = attack_img_flatten.reshape_as(img_tensor)
            attack_img = torch.clamp(attack_img, 0.0, 1.0)
            
            with torch.no_grad():
                logits = self.model(attack_img.to(self.device))
                curr_pred = torch.argmax(logits[0]).item()
            i += 1
        
        if (curr_pred == target_label):
            print('Y')
        return attack_img
            
    def _get_jacob(
        self, 
        img_tensor: torch.Tensor
    ) -> torch.Tensor:
        img_clone = img_tensor.clone().detach()
        img_clone.requires_grad_(True)
        out_logits = self.model(img_clone)
        
        tot_feat_num = int(np.prod(img_tensor.shape[1:]))
        jacob = torch.zeros((out_logits.shape[1], tot_feat_num))
        
        for i in range(out_logits.shape[1]):
            if img_clone.grad is not None:
                img_clone.grad.zero_()
            out_logits[0][i].backward(retain_graph = True)
            jacob[i] = img_clone.grad.reshape(-1).clone()
        return jacob
    
    def saliency_map(
        self,
        jacob: torch.Tensor, 
        target_label: int,
        search_domain: torch.Tensor
    ) -> Tuple[int, int]:
        tot_grad = torch.sum(jacob, dim = 0).to(self.device)
        target_grad = jacob[target_label].to(self.device)
        others_grad = tot_grad - target_grad
        
        max_num, p1, p2 = 0, 0, 0
        for p in range(len(target_grad) - 1):
            if search_domain[p] == False:
                continue
            alphas = target_grad[p] + target_grad[p+1:]
            betas = others_grad[p] + others_grad[p+1:]
            alphas[alphas <= 0] = 0
            betas[betas >= 0] = 0
            tmp = -alphas * betas * search_domain[p+1:].float().to(self.device)
            tmp_max = tmp.max()
            
            if tmp.max() > max_num:
                max_num = tmp_max
                tmp_max_idx = tmp.argmax()
                p1 = p
                p2 = (p + tmp_max_idx + 1).item()
        return int(p1), int(p2)
            