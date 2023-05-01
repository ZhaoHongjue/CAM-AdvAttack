import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .base import BaseCAM

class SSCAM(BaseCAM):
    def __init__(
        self, 
        model: nn.Module, 
        dataset: str,
        target_layer: str, 
        fc_layer: str = None, 
        use_relu: bool = False, 
        use_cuda: bool = True,
        smooth_mode: str = 'act'
    ) -> None:
        super().__init__(
            model, dataset, target_layer, fc_layer, use_relu, use_cuda
        )
        self.smooth_mode = smooth_mode
        self.use_relu = True
        
    def _get_raw_heatmap(self, img_tensor: torch.Tensor) -> torch.Tensor:
        n = 30
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim = 1)
            pred_idx = probs.argmax().item()
            
            featuremaps: torch.Tensor = self._get_feature_maps(img_tensor).squeeze(0)
            tfm = transforms.Resize(224)
            upsample_featuremaps = tfm(featuremaps)
            H = self.normalize(upsample_featuremaps)
            
            baseline = torch.zeros_like(img_tensor).to(self.device)
            tmp2 = self.model(baseline)
            
            cic_tot = torch.zeros(featuremaps.shape[0], logits.shape[1]).to(self.device)
            for _ in range(n):
                if self.smooth_mode == 'act':
                    H_noise = H + torch.normal(
                        mean = 0, std = 0.2, size = H.shape
                    ).to(self.device)
                    M = img_tensor * H_noise.unsqueeze(1)
                    cic_tot += self.model(M) - tmp2
                elif self.smooth_mode == 'input':
                    M = img_tensor * H.unsqueeze(1)
                    M += torch.normal(
                        mean = 0, std = 0.2, size = H.shape
                    ).to(self.device)
                    cic_tot += self.model(M) - tmp2
            
            cic_mean = cic_tot / n
            weights = F.softmax(cic_mean[:, pred_idx], dim = 0).reshape(-1, 1, 1)
            return (weights * featuremaps).sum(dim = 0)