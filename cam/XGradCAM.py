import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseCAM

class XGradCAM(BaseCAM):
    def __init__(
        self, 
        model: nn.Module, 
        dataset: str,
        target_layer: str, 
        fc_layer: str = None, 
        cuda: int = None
    ) -> None:
        super().__init__(
            model, dataset, target_layer, fc_layer, cuda
        )
        
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        grads = self._get_grads(img_normalized, pred, use_softmax = False)
        num = grads * self.featuremaps
        den = torch.sum(self.featuremaps, dim = (-1, -2), keepdim = True) + 1e-8
        weights = torch.sum(num / den, dim = (-1, -2), keepdim = True)
        return (weights * self.featuremaps).sum(dim = 1)
        # saliency_maps = []
        # for i in range(len(img_normalized)):
        #     grads = self._get_grads(img_normalized[i].unsqueeze(0), use_softmax = False)
            
        #     num = grads * self.featuremaps[i]
        #     den = torch.sum(
        #         self.featuremaps[i], dim = (-1, -2), keepdim = True
        #     ) + 1e-5
            
        #     weights = torch.sum(
        #         num / den, dim = (-1, -2), keepdim = True
        #     )
        #     saliency_maps.append((weights * self.featuremaps[i]).sum(dim = 0))
        # return torch.cat([s.unsqueeze(0) for s in saliency_maps])