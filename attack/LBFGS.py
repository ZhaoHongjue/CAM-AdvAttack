import torch
from torch import nn
from scipy.optimize import fmin_l_bfgs_b as lbfgsb
# from ._lbfgs import fmin_l_bfgs_b as lbfgsb
import numpy as np

from .base import BaseAttack

class LBFGS(BaseAttack):
    '''
    Paper: Intriguing properties of neural networks
    
    URL: http://arxiv.org/abs/1312.6199
    '''
    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def predict(
        self, 
        perturb: np.ndarray, 
        img_tensor: torch.Tensor,
        target_cls: int,
        c: float
    ):
        img_clone = img_tensor.clone().detach().cpu() 
        if img_clone.dim() == 3:
            img_clone = img_clone.unsqueeze(0)
            
        perturb: torch.Tensor = torch.tensor(
            perturb, dtype = torch.float32
        ).reshape_as(img_clone)
        perturb.requires_grad_(True)
        img_clone = img_clone + perturb
        
        self.model.zero_grad()
        img_clone = img_clone.to(self.device)
        Y_hat = self.model(img_clone)
        Y_label = torch.tensor(
            [target_cls], dtype = torch.long
        ).to(self.device)
        loss: torch.Tensor = c*torch.linalg.norm(perturb) + self.loss_fn(Y_hat, Y_label)
        loss.backward()
        grad: np.ndarray = perturb.grad.clone().numpy().reshape(-1).astype(np.float64)
        return loss.item(), grad
 
    def __call__(
        self, 
        img_tensor: torch.Tensor,
        target_cls: int,
        c: float = 0.1
    ) -> torch.Tensor:
        pixel_num = np.prod(img_tensor.shape)
        img_pixes = img_tensor.reshape(-1, 1).cpu().numpy()
        bounds_np = np.array([[0, 1]]*pixel_num) - img_pixes
        x0 = np.zeros((pixel_num,), dtype = np.float32)
        opt_func = lambda x: self.predict(x, img_tensor, target_cls, c)
        perturb, f, info = lbfgsb(
			func = opt_func,
			x0 = x0,
			bounds = bounds_np.tolist(),
		)
        print(perturb)
        perturb = torch.tensor(
            perturb, dtype = torch.float32
        ).reshape_as(img_tensor)
        perturb_img = img_tensor.cpu() + perturb.cpu()
        return perturb_img.detach()