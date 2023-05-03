import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseCAM

class GradCAM(BaseCAM):
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
        
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        saliency_maps = []
        for i in range(len(img_normalized)):
            grad = self._get_grads(img_normalized[i].unsqueeze(0), use_softmax = False)
            weights = torch.mean(grad, dim = (-1, -2), keepdim = True)
            saliency_map = (weights * self.featuremaps[i]).sum(dim = 0)
            saliency_maps.append(saliency_map)
        return torch.cat([s.unsqueeze(0) for s in saliency_maps])