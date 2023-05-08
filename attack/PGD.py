import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseAttack
from tqdm import trange

class PGD(BaseAttack):
    '''
    Paper: Towards Deep Learning Models Resistant to Adversarial
    
    URL: http://arxiv.org/abs/1706.06083
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
        img_tensor: torch.Tensor,
        label: int, 
        max_iter: int = 10,
        alpha: float = 2/255,
        eps: float = 0.2,
        **kwargs
    ) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss()
        img_clone = img_tensor.clone().detach().to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
        raw_img = img_clone.clone().detach().to(self.device)
        
        for _ in range(max_iter):
            self.model.zero_grad()
            img_clone.requires_grad_(True)
            Y_pred = self.model(self.tfm(img_clone))
            Y_label = torch.tensor(
                [label], dtype = torch.long
            ).to(self.device)
            loss: torch.Tensor = loss_fn(Y_pred, Y_label)
            loss.backward()
            
            grad = img_clone.grad
            with torch.no_grad():
                img_clone = img_clone + alpha * (grad.sign())
                adv = img_clone - raw_img
                delta = torch.clamp(adv, -eps, eps)
                img_clone = torch.clamp(raw_img + delta, 0, 1)
        return img_clone.cpu().detach()