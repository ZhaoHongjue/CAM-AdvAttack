import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseAttack
from tqdm import trange

class DeepFool(BaseAttack):
    '''
    Paper: DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks
    
    URL: https://openaccess.thecvf.com/content_cvpr_2016/html/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.html
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
        labels: torch.Tensor = None,
        max_iter: int = 100,
        num_classes: int = 10,
        attack_kwargs: dict = {}
    ) -> torch.Tensor:
        att_imgs = torch.zeros_like(imgs)
        with trange(len(imgs)) as t:
            for i in t:
                att_imgs[i] = self.attack_one(imgs[i], num_classes, max_iter)
        return att_imgs
    
    def attack_one(
        self, 
        img: torch.Tensor,
        num_classes: int,
        max_iter: int = 100,
        **kwargs
    ) -> torch.Tensor:
        img_clone, i = img.clone(), 0
        img_clone = img_clone.to(self.device)
        if img_clone.dim() == 3:
            img_clone.unsqueeze_(0)
            
        with torch.no_grad():
            raw_logits = self.model(self.tfm(img_clone))
            label = torch.argmax(F.softmax(raw_logits, dim = 1)).item()
        perturbed_label = label
        
        while i < max_iter:
            grads, fs = [], []
            self.model.zero_grad()
            img_clone.requires_grad_(True)
            if img_clone.grad is not None:
                img_clone.grad.zero_()
            logits: torch.Tensor = self.model(self.tfm(img_clone))
            with torch.no_grad():
                perturbed_label = F.softmax(logits, dim = 1).argmax().item()
                # print(f'perturbed_label: {perturbed_label}, label: {label}, i: {i}')
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
            with torch.no_grad():
                img_clone += r 
                img_clone = torch.clamp(img_clone, 0, 1)
            i += 1
            
        return img_clone.detach().cpu()