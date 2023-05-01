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
        use_relu: bool = False, 
        use_cuda: bool = True
    ) -> None:
        super().__init__(
            model, dataset, target_layer, fc_layer, use_relu, use_cuda
        )
        
    def _get_raw_heatmap(self, img_tensor: torch.Tensor) -> torch.Tensor:
        featuremaps: torch.Tensor = self._get_feature_maps(img_tensor).squeeze(0)
        grads = self._get_grads(img_tensor, use_softmax = True)
        grads2, grads3 = grads**2, grads**3
        a = grads2 / (2 * grads2 + torch.sum(
            grads3 * featuremaps, dim = (-1, -2), keepdim = True
        ))
        weights = torch.sum(a * F.relu(grads), dim = (-1, -2), keepdim = True)
        return (weights * featuremaps).sum(dim = 0)
