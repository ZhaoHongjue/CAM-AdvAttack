import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .base import BaseCAM


class ScoreCAM(BaseCAM):
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
        self.use_relu = True
        
    @torch.no_grad()
    def _get_cic(self, img_tensor: torch.Tensor, upsample_featuremaps: torch.Tensor) -> torch.Tensor:
        H = self.normalize(upsample_featuremaps)
        mask_imgs = (img_tensor * H.unsqueeze(1)).to(self.device)
        baseline = torch.zeros_like(img_tensor).to(self.device)
        self.model.to(self.device)
        cic = self.model(mask_imgs) - self.model(baseline)
        return cic
    
    def _get_raw_heatmap(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            featuremaps: torch.Tensor = self._get_feature_maps(img_tensor).squeeze(0)
            tfm = transforms.Resize(224)
            upsample_featuremaps = tfm(featuremaps)
            
            cic = self._get_cic(img_tensor, upsample_featuremaps)
            
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim = 1)
            pred_idx = probs.argmax().item()
            
            weights = F.softmax(cic[:, pred_idx], dim = 0).reshape(-1, 1, 1)
            return (weights * featuremaps).sum(dim = 0)
            
        