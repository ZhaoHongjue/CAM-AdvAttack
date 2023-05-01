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
        use_cuda: bool = True
    ) -> None:
        super().__init__(
            model, dataset, target_layer, fc_layer, use_relu, use_cuda
        )
        
    def _get_raw_heatmap(self, img_tensor: torch.Tensor) -> torch.Tensor:
        grads = self._get_grads(img_tensor, use_softmax = False)
        weights = torch.mean(grads, dim = (-1, -2), keepdim = True)
        featuremaps: torch.Tensor = self._get_feature_maps(img_tensor).squeeze(0)
        return (weights * featuremaps).sum(dim = 0)



        