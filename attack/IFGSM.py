import torch
from torch import nn

from .base import BaseAttack

class IFGSM(BaseAttack):
    '''
    Adversarial examples in the physical world
    '''
    def __init__(self, model: nn.Module, cuda: int = None) -> None:
        super().__init__(model, cuda)
    
    def __call__(
        self, 
        img_tensor: torch.Tensor,
        label: int, 
        max_iter: int = 5,
        eps: float = 0.1,
    ) -> torch.Tensor:
        loss_fn = nn.CrossEntropyLoss()
        img_clone = img_tensor.clone().detach().to(self.device)
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
            delta = eps / (k+1) * (grad.sign()).reshape_as(img_tensor)
            img_clone = img_clone + delta
            img_clone = torch.clamp(img_clone, 0, 1)
        return img_clone.cpu().detach()