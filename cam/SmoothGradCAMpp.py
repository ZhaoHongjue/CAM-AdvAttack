import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseCAM

class SmoothGradCAMpp(BaseCAM):
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
        grads_lst = [[], [], []]
        n = 100
        featuremaps = self._get_feature_maps(img_tensor).squeeze(0)

        for _ in range(n):
            img_tensor_noise = img_tensor + torch.normal(
                mean = 0, std = 0.5, size = img_tensor.shape
            ).to(self.device)
            grads = self._get_grads(img_tensor_noise, use_softmax = True)
            for i in range(3):
                grads_lst[i].append(grads**(i+1))
              
        Ds = [sum(grads_lst[i]) / n for i in range(3)]
        
        a = Ds[0] / (2 * Ds[1] + torch.sum(
            Ds[2] * featuremaps, dim = (-1, -2), keepdim = True
        ))
        weights = torch.sum(a * F.relu(Ds[0]), dim = (-1, -2), keepdim = True)
        return (weights * featuremaps).sum(dim = 0)
            
            
        