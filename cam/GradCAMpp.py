import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseCAM

class GradCAMpp(BaseCAM):
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
        self.use_relu = True
        
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        grads = self._get_grads(img_normalized, pred, use_softmax = False)
        grads2, grads3 = grads**2, grads**3
        den = 2 * grads2 + torch.sum(
            grads3 * self.featuremaps, dim = (-1, -2), keepdim = True
        )
        a = grads2 / (den + 1e-9)
        weights = torch.sum(a * F.relu(grads), dim = (-1, -2), keepdim = True)
        return (weights * self.featuremaps).sum(dim = 1)
        # saliency_maps = []
        # for i in range(len(img_normalized)):
        #     grads = self._get_grads(img_normalized[i].unsqueeze(0), pred[i], use_softmax = True)
        #     grads2, grads3 = grads**2, grads**3
        #     den = 2 * grads2 + torch.sum(
        #         grads3 * self.featuremaps[i], dim = (-1, -2), keepdim = True
        #     )
        #     a = grads2 / (den + 1e-8)
        #     weights = torch.sum(a * F.relu(grads), dim = (-1, -2), keepdim = True)
        #     saliency_map = (weights * self.featuremaps[i]).sum(dim = 0)
        #     saliency_maps.append(saliency_map)
        # return torch.cat([s.unsqueeze(0) for s in saliency_maps])