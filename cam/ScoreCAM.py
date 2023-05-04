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
        cuda: int = None
    ) -> None:
        super().__init__(
            model, dataset, target_layer, fc_layer, use_relu, cuda
        )
        self.use_relu = True
        
    @torch.no_grad()
    def _get_cic(self, img: torch.Tensor, upsample_featuremaps: torch.Tensor) -> torch.Tensor:
        H = self.normalize_featuremaps(upsample_featuremaps)
        mask_imgs = img.unsqueeze(1) * H.unsqueeze(2)
        baseline = torch.zeros_like(img[0].unsqueeze(0)).to(self.device)
        self.model.to(self.device)
        cics = []
        baseline_out = self.model(baseline)
        for i in range(len(img)):
            cic = (self.model(mask_imgs[i].to(self.device)) - baseline_out).cpu()
            cics.append(cic)
        return torch.cat([cic.unsqueeze(0) for cic in cics])
    
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        upsample_featuremaps = transforms.Resize(img_normalized.shape[-1])(self.featuremaps).cpu()
        cics = self._get_cic(img_normalized, upsample_featuremaps)
        indices1 = torch.arange(len(pred)).reshape(-1, 1)
        indices2 = pred.reshape(-1, 1)
        weights = F.softmax(cics[indices1, :, indices2].squeeze(1), dim = 1).unsqueeze(-1).unsqueeze(-1)
        return (weights.to(self.device) * self.featuremaps).sum(dim = 1)
        # with torch.no_grad():
        #     upsample_featuremaps = transforms.Resize(img_normalized.shape[-1])(self.featuremaps)
        #     H = self.normalize_featuremaps(upsample_featuremaps)
        #     mask_imgs = img_normalized.unsqueeze(1).to(self.device) * H.unsqueeze(2).to(self.device)
        #     baseline = torch.zeros_like(img_normalized[0].unsqueeze(0)).to(self.device)
        #     self.model.to(self.device)
        #     baseline_out = self.model(baseline)
        #     saliency_maps = []
        #     for i in range(len(img_normalized)):
        #         cic = self.model(mask_imgs[i]) - baseline_out
        #         weights = F.softmax(cic[:, pred[i]], dim = 0).reshape(-1, 1, 1)
        #         saliency_maps.append((weights * self.featuremaps[i]).sum(dim = 0).cpu())
        # return torch.cat([s.unsqueeze(0) for s in saliency_maps])