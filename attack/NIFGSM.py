import torch
from torch import nn

from .base import BaseAttack
from tqdm import trange

class NIFGSM(BaseAttack):
    '''
    Boosting Adversarial Attacks With Momentum
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
        max_iter: int = 5,
        eps: float = 0.1,
        mu: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        alpha = eps / max_iter
        loss_fn = nn.CrossEntropyLoss()
        img_clone = img_tensor.clone().detach().to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
        g = torch.zeros_like(img_tensor).to(self.device)
        
        for k in range(max_iter):
            if k != 0:
                img_clone = img_clone.clone().detach()
            
            img_nes = img_clone + alpha * mu * g
            img_nes.requires_grad_(True)
            
            self.model.zero_grad()
            Y_pred = self.model(self.tfm(img_nes))
            Y_label = torch.tensor([label], dtype = torch.long).to(self.device)
            loss = loss_fn(Y_pred, Y_label)
            loss.backward()
            
            grad = img_nes.grad.reshape_as(img_tensor)
            g = mu * g + grad / torch.linalg.norm(grad)
            
            delta = alpha * (g.sign())
            img_clone = torch.clamp(img_clone + delta, 0, 1)
        return img_clone.cpu().detach()