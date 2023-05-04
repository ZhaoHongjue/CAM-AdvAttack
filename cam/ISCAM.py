import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .base import BaseCAM


class ISCAM(BaseCAM):
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
        
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        n = 10
        with torch.no_grad():
            upsample_featuremaps = transforms.Resize(img_normalized.shape[-1])(self.featuremaps).cpu()
            H = self.normalize_featuremaps(upsample_featuremaps)
            mask_imgs = img_normalized.unsqueeze(1) * H.unsqueeze(2)
            
            baseline = torch.zeros_like(img_normalized[0].unsqueeze(0)).to(self.device)
            self.model.to(self.device)
            baseline_out = self.model(baseline)
            cic_tot = torch.zeros((
                H.shape[0], H.shape[1], baseline_out.shape[1]
            ))
            M = torch.zeros_like(mask_imgs)
            for i in range(n):
                M = M + mask_imgs * i / n
                for j in range(len(cic_tot)):
                    cic_tot[j] += (self.model(M[j].to(self.device)) - baseline_out).cpu()
            cic_mean = cic_tot / n
                
            indices1 = torch.arange(len(pred)).reshape(-1, 1)
            indices2 = pred.reshape(-1, 1)
            weights = F.softmax(
                cic_mean[indices1, :, indices2].squeeze(1), 
                dim = 1
            ).unsqueeze(-1).unsqueeze(-1)
            return (weights.to(self.device) * self.featuremaps).sum(dim = 1)
            # saliency_maps = []
            
            # for i in range(len(img_normalized)):
            #     cic_tot = torch.zeros(
            #         self.featuremaps[i].shape[0], baseline_out.shape[1]
            #     ).to(self.device)
            #     M = torch.zeros_like(mask_imgs[i])
            #     for j in range(n):
            #         M = M + mask_imgs[i] * j / n
            #         cic_tot += self.model(M) - baseline_out
            #     cic_mean = cic_tot / n
            #     weights = F.softmax(cic_mean[:, pred[i]], dim = 0).reshape(-1, 1, 1)
            #     saliency_maps.append((weights * self.featuremaps[i]).sum(dim = 0)) 
            # return torch.cat([s.unsqueeze(0) for s in saliency_maps])