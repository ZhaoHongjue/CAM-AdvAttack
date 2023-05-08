import torch
from torch import nn

from .base import BaseAttack
from tqdm import trange


class CW(BaseAttack):
    def __init__(
        self, 
        model: nn.Module,
        dataset: str, 
        cuda: int = None
    ) -> None:
        super().__init__(model, dataset, cuda)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def __call__(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 200,
        num_classes: int = None,
        attack_kwargs: dict = {}
    ) -> torch.Tensor:
        target_clses = (labels + 1) % 10
        att_imgs = torch.zeros_like(imgs)
        with trange(len(imgs)) as t:
            for i in t:
                att_imgs[i] = self.attack_one(
                    imgs[i], target_clses[i], max_iter, **attack_kwargs
                )
        return att_imgs
    
    def f1(self, img_in, target_cls):
        logits = self.model(self.tfm(img_in.unsqueeze(0)).to(self.device))
        target_cls = torch.tensor([target_cls], dtype = torch.long).to(self.device)
        # return (1 - self.loss_fn(logits, target_cls))
        return self.loss_fn(logits, target_cls)
        
    def opt_func(
        self, 
        img: torch.Tensor, 
        delta: torch.Tensor, 
        target_cls: int,
        c: float
    ):
        img_in = torch.clamp(img + delta, 0, 1)
        loss = delta.norm() + c * self.f1(img_in, target_cls)
        return loss 

    def attack_one(
        self, 
        img: torch.Tensor,
        target_cls: int, 
        max_iter: int = 200,
        c: float = 0.05,
        lr: float = 0.01,
        **kwargs
    ) -> torch.Tensor:
        self.model.to(self.device)
        delta = torch.zeros_like(img, requires_grad = True)
        opt = torch.optim.Adam([delta], lr = lr)
        
        for idx in range(max_iter):
            opt.zero_grad()
            loss = self.opt_func(img, delta, target_cls, c)
            loss.backward()
            # print(loss.item())
            opt.step()
        
        perturb_img = torch.clamp(img + delta.detach(), 0, 1)
        return perturb_img