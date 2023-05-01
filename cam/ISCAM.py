import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .base import BaseCAM


class ISCAM(BaseCAM):
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: str, 
        fc_layer: str = None, 
        use_relu: bool = False, 
        use_cuda: bool = True
    ) -> None:
        super().__init__(model, target_layer, fc_layer, use_relu, use_cuda)
        self.use_relu = True
        
    def _get_raw_heatmap(self, img_tensor: torch.Tensor) -> torch.Tensor:
        n = 10
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim = 1)
            pred_idx = probs.argmax().item()
            
            featuremaps: torch.Tensor = self._get_feature_maps(img_tensor).squeeze(0)
            tfm = transforms.Resize(224)
            upsample_featuremaps = tfm(featuremaps)
            H = self.normalize(upsample_featuremaps)
            mask_imgs = img_tensor * H.unsqueeze(1)
            
            baseline = torch.zeros_like(img_tensor).to(self.device)
            tmp2 = self.model(baseline)
            
            cic_tot = torch.zeros(featuremaps.shape[0], logits.shape[1]).to(self.device)
            M = torch.zeros_like(mask_imgs)
            for i in range(n):
                M = M + mask_imgs * i / n
                cic_tot += self.model(M) - tmp2
                
            cic_mean = cic_tot / n
            weights = F.softmax(cic_mean[:, pred_idx], dim = 0).reshape(-1, 1, 1)
            return (weights * featuremaps).sum(dim = 0)