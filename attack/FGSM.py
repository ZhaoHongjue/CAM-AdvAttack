import torch
from torch import nn

from .base import BaseAttack
from tqdm import trange


class FGSM(BaseAttack):
    def __init__(self, model: nn.Module, cuda: int = None) -> None:
        super().__init__(model, cuda)
    
    def __call__(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = None,
        num_classes: int = None,
        attack_kwargs: dict = {}
    ) -> torch.Tensor:
        att_imgs = torch.zeros_like(imgs)
        with trange(len(imgs)) as t:
            for i in t:
                att_imgs[i] = self.attack_one(
                    imgs[i], labels[i], **attack_kwargs
                )
        return att_imgs
    
    def attack_one(
        self, 
        img: torch.Tensor,
        label: int, 
        eps: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        self.model.zero_grad()
        loss_fn = nn.CrossEntropyLoss()
        img_clone = img.clone().detach().to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
        img_clone.requires_grad_(True)
        
        Y_pred = self.model(img_clone)
        Y_label = torch.tensor([label], dtype = torch.long).to(self.device)
        loss = loss_fn(Y_pred, Y_label)
        loss.backward()
        
        grad = img_clone.grad
        delta = eps * (grad.sign()).reshape_as(img).cpu()
        perturbed_img = img.cpu() + delta
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img.cpu().detach()