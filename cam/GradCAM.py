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
        weights = torch.mean(grads, dim = (-1, -2), keepdim = True)
        return (weights * self.featuremaps).sum(dim = 1)