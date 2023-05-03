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
        cuda: int = None,
        smooth_mode: str = 'act'
    ) -> None:
        super().__init__(
            model, dataset, target_layer, fc_layer, use_relu, cuda
        )
        self.smooth_mode = smooth_mode
        self.use_relu = True
        
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        n = 30
        with torch.no_grad():
            baseline = torch.zeros_like(img_normalized[0].unsqueeze(0)).to(self.device)
            self.model.to(self.device)
            baseline_out = self.model(baseline)
            
            upsample_featuremaps = transforms.Resize(img_normalized.shape[-1])(self.featuremaps)
            H = self.normalize_featuremaps(upsample_featuremaps)
        
            saliency_maps = []
            for i in range(len(img_normalized)):
                cic_tot = torch.zeros((H.shape[1], baseline_out.shape[1])).to(self.device)
                for _ in range(n):
                    if self.smooth_mode == 'act':
                        H_noise = H[i] + torch.normal(
                            mean = 0, std = 0.2, size = H[i].shape
                        ).to(self.device)
                        M = img_normalized[i].to(self.device) * H_noise.unsqueeze(1)
                        cic_tot += self.model(M) - baseline_out
                    elif self.smooth_mode == 'input':
                        M = img_normalized[i] * H[i].unsqueeze(1)
                        M += torch.normal(
                            mean = 0, std = 0.2, size = H[i].shape
                        ).to(self.device)
                        cic_tot += self.model(M) - baseline_out
                
                cic_mean = cic_tot / n
                weights = F.softmax(cic_mean[:, pred[i]], dim = 0).reshape(-1, 1, 1)
                saliency_maps.append((weights * self.featuremaps[i]).sum(dim = 0).cpu())
            return torch.cat([s.unsqueeze(0) for s in saliency_maps])