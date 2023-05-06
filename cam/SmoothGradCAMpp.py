import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseCAM

class SMGradCAMpp(BaseCAM):
    def __init__(
        self, 
        model: nn.Module, 
        dataset: str,
        target_layer: str, 
        fc_layer: str = None, 
        use_relu: bool = False, 
        cuda: int = None
    ) -> None:
        super().__init__(
            model, dataset, target_layer, fc_layer, use_relu, cuda
        )
        self.use_relu = True
        
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        n = 100
        grads_lst = [[], [], []]
        for _ in range(n):
            img_noise = img_normalized.to(self.device) + torch.normal(
                mean = 0, std = 0.5, size = img_normalized.shape
            ).to(self.device)
            grads = self._get_grads(
                img_noise, pred, use_softmax = True
            )
            for i in range(3):
                grads_lst[i].append(grads**(i+1))
        
        Ds = [sum(grads_lst[i]) / n for i in range(3)]
        den = 2 * Ds[1] + torch.sum(
            Ds[2] * self.featuremaps[i], dim = (-1, -2), keepdim = True
        ) 
        a = Ds[0] / (den)
        weights = torch.sum(a * F.relu(Ds[0]), dim = (-1, -2), keepdim = True)
        return (weights * self.featuremaps).sum(dim = 1)
        # saliency_maps = []
        # for i in range(len(img_normalized)):
        #     grads_lst = [[], [], []]
        #     n = 100

        #     for _ in range(n):
        #         img_noise = img_normalized[i].to(self.device) \
        #             + torch.normal(
        #             mean = 0, std = 0.5, size = img_normalized[i].shape
        #         ).to(self.device)
        #         grads = self._get_grads(
        #             img_noise.unsqueeze(0), use_softmax = True
        #         )
        #         for i in range(3):
        #             grads_lst[i].append(grads**(i+1))
                
        #     Ds = [sum(grads_lst[i]) / n for i in range(3)]
            
            
        #     weights = torch.sum(a * F.relu(Ds[0]), dim = (-1, -2), keepdim = True)
        #     saliency_map = (weights * self.featuremaps[i]).sum(dim = 0)
        #     saliency_maps.append(saliency_map)
        # return torch.cat([s.unsqueeze(0) for s in saliency_maps])
            
            
        