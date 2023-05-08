import torch
from torch import nn

from .base import BaseAttack
from tqdm import trange
from math import ceil

class IFGSM(BaseAttack):
    '''
    Adversarial examples in the physical world
    '''
    def __init__(self, model: nn.Module, cuda: int = None) -> None:
        super().__init__(model, cuda)
    
    def __call__(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 10,
        num_classes: int = None,
        attack_kwargs: dict = {}
    ) -> torch.Tensor:
        att_imgs = torch.zeros_like(imgs)
        with trange(len(imgs)) as t:
            for i in t:
                att_imgs[i] = self.attack_one(
                    imgs[i], labels[i], max_iter, **attack_kwargs
                )
        return att_imgs
    
    def attack_one(
        self, 
        img: torch.Tensor,
        label: int, 
        max_iter: int,
        eps: float = 0.1,
    ) -> torch.Tensor:
        alpha = eps / max_iter
        loss_fn = nn.CrossEntropyLoss()
        img_clone = img.clone().detach().to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
        
        for k in range(max_iter):
            if k != 0:
                img_clone = img_clone.clone().detach()
            img_clone.requires_grad_(True)
            
            self.model.zero_grad()
            Y_pred = self.model(img_clone)
            Y_label = torch.tensor([label], dtype = torch.long).to(self.device)
            loss = loss_fn(Y_pred, Y_label)
            loss.backward()
            
            grad = img_clone.grad
            delta = alpha * (grad.sign()).reshape_as(img)
            img_clone = img_clone + delta
            img_clone = torch.clamp(img_clone, 0, 1)
        return img_clone.cpu().detach()