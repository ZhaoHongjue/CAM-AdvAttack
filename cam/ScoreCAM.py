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
        H = self.normalize_featuremaps(upsample_featuremaps)
        mask_imgs = img_tensor.unsqueeze(1).to(self.device) * H.unsqueeze(2).to(self.device)
        baseline = torch.zeros_like(img_tensor[0].unsqueeze(0)).to(self.device)
        self.model.to(self.device)
        cics = []
        baseline_out = self.model(baseline)
        for i in range(len(img_tensor)):
            # print('1', self.model(mask_imgs[i]).shape, baseline_out.shape)
            cic = self.model(mask_imgs[i]) - baseline_out
            cics.append(cic)
        return torch.cat([cic.unsqueeze(0) for cic in cics])
    
    def _get_raw_saliency_map(self, img_tensor: torch.Tensor) -> torch.Tensor:
        preds, _ = self.model_predict(img_tensor)
        upsample_featuremaps = transforms.Resize(img_tensor.shape[-1])(self.featuremaps)
        H = self.normalize_featuremaps(upsample_featuremaps)
        mask_imgs = img_tensor.unsqueeze(1).to(self.device) * H.unsqueeze(2).to(self.device)
        baseline = torch.zeros_like(img_tensor[0].unsqueeze(0)).to(self.device)
        self.model.to(self.device)
        baseline_out = self.model(baseline)
        saliency_maps = []
        for i in range(len(img_tensor)):
            cic = self.model(mask_imgs[i]) - baseline_out
            weights = F.softmax(cic[:, preds[i]], dim = 0).reshape(-1, 1, 1)
            saliency_maps.append((weights * self.featuremaps[i]).sum(dim = 0).cpu().detach())
        return torch.cat([s.unsqueeze(0) for s in saliency_maps])
    

        # with torch.no_grad():
        #     featuremaps: torch.Tensor = self._get_feature_maps(img_tensor).squeeze(0)
        #     tfm = transforms.Resize(224)
        #     upsample_featuremaps = tfm(featuremaps)
            
        #     cic = self._get_cic(img_tensor, upsample_featuremaps)
            
        #     logits = self.model(img_tensor)
        #     probs = F.softmax(logits, dim = 1)
        #     pred_idx = probs.argmax().item()
            
        #     weights = F.softmax(cic[:, pred_idx], dim = 0).reshape(-1, 1, 1)
        #     return (weights * featuremaps).sum(dim = 0)
            
        