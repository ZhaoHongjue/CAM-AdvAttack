import torch
from torch import nn

from .base import BaseAttack

class FGSM(BaseAttack):
    def __init__(self, model: nn.Module, cuda: int = None) -> None:
        super().__init__(model, cuda)
    
    def __call__(
        self, 
        img_tensor: torch.Tensor,
        label: int, 
        eps: float = 0.1,
    ) -> torch.Tensor:
        self.model.zero_grad()
        loss_fn = nn.CrossEntropyLoss()
        img_clone = img_tensor.clone().detach().to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
        img_clone.requires_grad_(True)
        
        Y_pred = self.model(img_clone)
        Y_label = torch.tensor([label], dtype = torch.long).to(self.device)
        loss = loss_fn(Y_pred, Y_label)
        loss.backward()
        
        grad = img_clone.grad
        delta = eps * (grad.sign()).reshape_as(img_tensor).cpu()
        perturbed_img = img_tensor.cpu() + delta
        perturbed_img = torch.clamp(perturbed_img, 0, 1)
        return perturbed_img.cpu().detach()