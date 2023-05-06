import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseCAM

class CAM(BaseCAM):
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
        if fc_layer is None:
            raise ValueError(
                '`fc_layer` can not be None!'
            )
        
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        fc: nn.Linear = self.model.get_submodule(self.fc_layer)
        weights =  fc.weight[pred].detach().reshape(len(pred), -1, 1, 1)
        return (weights * self.featuremaps).sum(dim = 1)