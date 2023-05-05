import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseAttack
from .DeepFool import DeepFool

class UniPerturb(BaseAttack):
    '''
    Paper: Universal Adversarial Perturbations
    
    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html
    '''
    def __init__(self, model: nn.Module, cuda: int = None) -> None:
        super().__init__(model, cuda)
             
    def project_perturb(
        self, 
        perturb: torch.Tensor,
        perturb_norm: float,
        norm_mode: str
    ) -> torch.Tensor:
        if norm_mode == 'Euc':
            raw_norm = torch.linalg.norm(perturb)
            if raw_norm > perturb_norm:
                perturb = perturb_norm * perturb / raw_norm
        else:
            raise ValueError(
                f'The norm mode can not be `{norm_mode}`'
            )
        return perturb
     
    def predict_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs_clone = imgs.clone().to(self.device)
        with torch.no_grad():
            pred = F.softmax(self.model(imgs_clone), dim = 1).argmax(dim = 1)
        return pred
        
    def __call__(
        self, 
        imgs: torch.Tensor,
        num_classes: int,
        step: int = 100,
        perturb_norm: float = 12.5,
        norm_mode: str = 'Euc',
        acc: float = 0.2,
        max_iter: int = 100,
    ) -> torch.Tensor:
        if imgs.dim() != 4:
            raise ValueError(
                'The dim of image tensor must be 4!'
            )
        imgs = imgs.to(self.device)
        perturb = torch.zeros(imgs.shape[1:]).to(self.device)
        raw_pred = self.predict_imgs(imgs)
        err_rate = 0
        
        i = 0
        while err_rate < 1 - acc and i <= max_iter:
            for img in imgs:
                single_perturb = self.calc_perturb(
                    img + perturb, num_classes, step
                ).to(self.device)
                # print(single_perturb.norm())
                perturb = self.project_perturb(
                    perturb + single_perturb, 
                    perturb_norm, 
                    norm_mode
                ).to(self.device)
            pred = self.predict_imgs(imgs + perturb)
            err_rate = (pred != raw_pred).sum().item() / len(pred)
            print(f'err rate {err_rate}')
            i+=1
        return perturb

    def calc_perturb(
        self,
        img_tensor: torch.Tensor,
        num_classes: int,
        step: int
    ) -> torch.Tensor:
        img_clone, i = img_tensor.clone(), 0
        img_clone = img_clone.to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
            
        with torch.no_grad():
            raw_logits = self.model(img_clone)
            label = torch.argmax(F.softmax(raw_logits, dim = 1)).item()
        perturbed_label = label
        
        rs = []
        while i < step:
            grads, fs = [], []
            self.model.zero_grad()
            img_clone.requires_grad_(True)
            if img_clone.grad is not None:
                img_clone.grad.zero_()
            logits: torch.Tensor = self.model(img_clone)
            with torch.no_grad():
                perturbed_label = F.softmax(logits, dim = 1).argmax().item()
                if perturbed_label != label:
                    break
            logits[0, label].backward(retain_graph = True)
            grad_raw = img_clone.grad.clone()
            
            for k in range(num_classes):
                if k == label: continue
                self.model.zero_grad()
                if img_clone.grad is not None:
                    img_clone.grad.zero_()
                logits[0, k].backward(retain_graph = True)
                grad = img_clone.grad.clone()
                
                grads.append(grad)
                fs.append(logits[0, k].item())
            
            tmp1 = (torch.as_tensor(fs) - logits[0, label].item()).to(self.device)
            tmp2 = (torch.cat(grads, dim = 0) - grad_raw).reshape(9, -1).to(self.device)
            tmp3: torch.Tensor = torch.abs(tmp1) / torch.linalg.norm(tmp2, dim = 1)
            l = tmp3.argmin().item()
            
            r = tmp3[l] * grads[l]
            rs.append(r)
            with torch.no_grad():
                img_clone += r 
                img_clone = torch.clamp(img_clone, 0, 1)
            i += 1
        return sum(r).cpu()