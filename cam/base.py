import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 7)
from PIL import Image
import cv2

from typing import List, Callable, Iterable, Dict, Tuple
from abc import abstractmethod

class_names = {
    'imagenette': (
        'tench', 'English springer', 'cassette player', 'chain saw', 'church', 
        'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
    ),
    'CIFAR10': (
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse',
        'Ship', 'Trunk'
    )
}

mean = {
    'FashionMNIST': (0,),
    'CIFAR10': (0.4914, 0.4822, 0.4465),
    'CIFAR100': (0.5071, 0.4867, 0.4408),
    'Imagenette': (0.485, 0.456, 0.406),
}

std = {
    'FashionMNIST': (1,),
    'CIFAR10': (0.2023, 0.1994, 0.2010),
    'CIFAR100': (0.2675, 0.2565, 0.2761),
    'Imagenette': (0.229, 0.224, 0.225),
}


class BaseCAM:
    def __init__(
        self, 
        model: nn.Module, 
        dataset: str,
        target_layer: str,
        fc_layer: str = None,
        use_relu: bool = False,
        use_cuda: bool = True
    ) -> None:
        # set basic attributes
        self.model = model
        self.tfm = transforms.Normalize(mean[dataset], std[dataset])
        self.use_relu = use_relu
        layer_names = self._get_layer_names()
        
        # Error detect
        if target_layer not in layer_names:
            raise AttributeError(
                f'Model has no attribute `{target_layer}`!'
            )
        
        if fc_layer:
            if fc_layer not in layer_names:
                raise AttributeError(
                    f'Model has no attribute `{fc_layer}`'
                )
            if not isinstance(
                self.model.get_submodule(fc_layer), 
                nn.Linear
            ):
                raise ValueError(
                    f'{fc_layer} is not a `nn.Linear` instance!'
                )
        
        if use_cuda:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device('cpu')
        
        self.model.to(self.device)
        
        self.fc_layer = fc_layer
        self.target_layer = target_layer
    
    def __call__(
        self, 
        img: torch.Tensor,
        mask_rate: float = 0.4, 
    ) -> Tuple[np.ndarray, int, float]:
        # Generate Numpy Image
        if img.dim() == 4:
            img_np = np.transpose(img.cpu().numpy(), (0, 2, 3, 1))
        elif img.dim() == 3:
            img_np = np.transpose(img.cpu().numpy(), (1, 2, 0))
        else:
            raise ValueError
        img_np = np.uint8(img_np * 255)
        
        img_in = self.tfm(img.clone().detach().cpu())
        if img_in.dim() == 3:
            img_in.unsqueeze_(0)
        
        self.features = []
        target_layer = dict([*self.model.named_children()])[self.target_layer]
        handle = target_layer.register_forward_hook(self.hook_featuremap())
        pred, prob = self.model_predict(img_in)
        handle.remove()
        self.featuremaps = self.features[0]
        
        saliency_map = self.generate_saliency_map(img_in, pred)
        
        heatmaps = []
        for i in range(len(saliency_map)):
            heatmap = cv2.applyColorMap(
                np.uint8(saliency_map[i] * 255),
                cv2.COLORMAP_JET
            )
            heatmaps.append(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        heatmaps = np.asarray(heatmaps)
        cam_np = np.uint8(img_np * (1 - mask_rate) + heatmaps * mask_rate)
        return cam_np, pred, prob
    
    @torch.no_grad()
    def model_predict(self, img: torch.Tensor):            
        self.model.to(self.device)
        probs = F.softmax(self.model(img.to(self.device)), dim = 1)
        max_info = probs.max(dim = 1)
        return max_info.indices.cpu(), max_info.values.cpu()
    
    def hook_featuremap(self):
        def hook_fn(_, __, output):
            self.features.append(output)
        return hook_fn
    
    def generate_saliency_map(
        self, 
        img: torch.Tensor,
        pred: torch.Tensor,
    ) -> Tuple[np.ndarray, int, float]:
        raw_saliency_map: torch.Tensor = self._get_raw_saliency_map(img, pred)
        if self.use_relu:
            raw_saliency_map = F.relu(raw_saliency_map)
        raw_max = torch.max(raw_saliency_map.reshape(len(img), -1), dim = 1).values.reshape(-1, 1, 1)
        raw_min = torch.min(raw_saliency_map.reshape(len(img), -1), dim = 1).values.reshape(-1, 1, 1)
        raw_saliency_map = (raw_saliency_map - raw_min) / (raw_max - raw_min)
        saliency_maps = transforms.Resize(img.shape[-1])(raw_saliency_map)
        return saliency_maps.cpu().numpy()
    
    @abstractmethod
    def _get_raw_saliency_map(
        self, 
        img: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def _get_layer_names(self) -> List[str]:
        layer_names = []
        for name, _ in self.model.named_children():
            layer_names.append(name)
        return layer_names
    
    def _get_grads(self, img_tensor: torch.Tensor, use_softmax: bool):
        grad_outs = []
        handle = self.model.get_submodule(self.target_layer).register_full_backward_hook(
            lambda _, __, go: grad_outs.append(go)
        )
        
        self.model.zero_grad()
        self.model.to(self.device)
        scores = self.model(img_tensor.clone().to(self.device))
        if use_softmax == 'logits':
            scores = F.softmax(scores, dim = 1)
        idx = scores.argmax(dim = 1)
        scores[:, idx].backward()
        
        handle.remove()
        return grad_outs[0][0].squeeze(0)
    
    def normalize_featuremaps(
        self, 
        upsample_featuremaps: torch.Tensor
    ) -> torch.Tensor:
        feat_reshape = upsample_featuremaps.reshape(upsample_featuremaps.shape[:2] + (-1,))
        maxs = feat_reshape.max(dim = -1).values.unsqueeze(-1).unsqueeze(-1)
        mins = feat_reshape.min(dim = -1).values.unsqueeze(-1).unsqueeze(-1)
        H = (upsample_featuremaps - mins) / (maxs - mins + 1e-5)
        return H

    
def plot_cam_img(
    cam_img_np: np.ndarray, dataset: str, class_idx: int, 
    prob: float, save_pth: str = None
) -> None:
    plt.clf()
    class_name = class_names[dataset][class_idx]
    plt.imshow(cam_img_np)
    if 0 <= prob <= 1:
        prob *= 100
    plt.title(f'{class_name} ({prob:4.2f}%)', fontsize = 22)
    plt.axis('off')
    
    if save_pth:
        plt.savefig(save_pth, bbox_inches = 'tight', pad_inches = 0.03)
    else:
        plt.show()