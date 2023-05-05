import torch
from torch import nn
import numpy as np

from .base import BaseAttack
from tqdm import trange


class IterLL(BaseAttack):
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
        num_classes: int = 10,
        attack_kwargs: dict = {}
    ) -> torch.Tensor:
        att_imgs = torch.zeros_like(imgs)
        with trange(len(imgs)) as t:
            for i in t:
                att_imgs[i] = self.attack_one(
                    imgs[i], num_classes, max_iter, **attack_kwargs
                )
        return att_imgs
    
    def attack_one(
        self, 
        img: torch.Tensor,
        num_classes: int, 
        max_iter: int = 5,
        eps: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss()
        img_clone = img.clone().detach().to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
        
        with torch.no_grad():
            logits = self.model(img_clone)
            losses = []
            for i in range(num_classes):
                Y_label = torch.tensor([i], dtype = torch.long).to(self.device)
                losses.append(float(loss_fn(logits, Y_label)))
            YLL = int(np.argmax(losses))
        Y_label = torch.tensor([YLL], dtype = torch.long).to(self.device)
        
        for k in range(max_iter):
            if k != 0:
                img_clone = img_clone.clone().detach()
            img_clone.requires_grad_(True)
            
            self.model.zero_grad()
            Y_pred = self.model(img_clone)
            loss = loss_fn(Y_pred, Y_label)
            loss.backward()
            
            grad = img_clone.grad
            delta = eps / (k+1) * (grad.sign()).reshape_as(img)
            img_clone = img_clone - delta
            img_clone = torch.clamp(img_clone, 0, 1)
        return img_clone.cpu().detach()