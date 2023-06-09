import torch
from torch import nn
import numpy as np

from .base import BaseAttack
from tqdm import trange


class StepLL(BaseAttack):
    '''
    Adversarial Machine Learning at Scale
    '''
    def __init__(
        self, 
        model: nn.Module,
        dataset: str, 
        cuda: int = None
    ) -> None:
        super().__init__(model, dataset, cuda)
        
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
                    imgs[i], num_classes, **attack_kwargs
                )
        return att_imgs
        
    def attack_one(
        self, 
        img_tensor: torch.Tensor,
        num_class: int, 
        eps: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss()
        img_clone = img_tensor.clone().detach().to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
        
        with torch.no_grad():
            logits = self.model(self.tfm(img_clone))
            losses = []
            for i in range(num_class):
                Y_label = torch.tensor([i], dtype = torch.long).to(self.device)
                losses.append(float(loss_fn(logits, Y_label)))
            YLL = int(np.argmax(losses))
        Y_label = torch.tensor([YLL], dtype = torch.long).to(self.device)
        
        img_clone.requires_grad_(True)
        self.model.zero_grad()
        Y_pred = self.model(self.tfm(img_clone))
        loss = loss_fn(Y_pred, Y_label)
        loss.backward()
        
        grad = img_clone.grad
        delta = eps * (grad.sign()).reshape_as(img_tensor)
        attack_img = img_clone - delta
        attack_img = torch.clamp(attack_img, 0, 1)

        return attack_img.cpu().detach()