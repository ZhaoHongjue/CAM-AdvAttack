import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseCAM

class CAM(BaseCAM):
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: str, 
        fc_layer: str = None, 
        use_relu: bool = False, 
        use_cuda: bool = True
    ) -> None:
        super().__init__(
            model, target_layer, fc_layer, use_relu, use_cuda
        )
        if fc_layer is None:
            raise ValueError(
                '`fc_layer` can not be None!'
            )
        
    def _get_raw_heatmap(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            featuremaps: torch.Tensor = self._get_feature_maps(img_tensor).squeeze(0)
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim = 1)
            pred_idx = probs.argmax().item()
            fc: nn.Linear = self.model.get_submodule(self.fc_layer)
            weights =  fc.weight[pred_idx].detach().reshape(-1, 1, 1)
            return (weights * featuremaps).sum(dim = 0)
            