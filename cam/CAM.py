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
        use_relu: bool = False, 
        use_cuda: bool = True
    ) -> None:
        super().__init__(
            model, dataset, target_layer, fc_layer, use_relu, use_cuda
        )
        if fc_layer is None:
            raise ValueError(
                '`fc_layer` can not be None!'
            )
        
    def _get_raw_saliency_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        pred, _ = self.model_predict(img_tensor)
        fc: nn.Linear = self.model.get_submodule(self.fc_layer)
        weights =  fc.weight[pred].detach().reshape(len(pred), -1, 1, 1)
        return (weights * self.featuremaps).sum(dim = 1)