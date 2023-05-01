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


class FeatureExtractor:
    def __init__(self, model: nn.Module, layers: Iterable[str]) -> None:
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}
        self.handles = []
        
        for layer_id in self.layers:
            layer = dict([*self.model.named_children()])[layer_id]
            self.handles.append(
                layer.register_forward_hook(self.hook_save_features(layer_id))
            )
            
    def hook_save_features(self, layer_id) -> Callable:
        def hook_fn(_, __, output):
            self._features[layer_id] = output
        return hook_fn
    
    def remove_hooks(self):
        for i in range(len(self.handles)):
            self.handles[i].remove()
    
    def __call__(self, X) -> Dict[str, torch.Tensor]:
        _ = self.model(X)
        return self._features


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
        self.model, self.dataset = model, dataset
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
        self.feature_extractor = FeatureExtractor(
            self.model, [target_layer]
        )
        
        self.fc_layer = fc_layer
        self.target_layer = target_layer
    
    def __call__(
        self, 
        img: Image.Image or torch.Tensor,
        mask_rate: float = 0.4, 
    ) -> Tuple[np.ndarray, int, float]:
        # process image
        # to np.ndarray
        if type(img) == Image.Image:
            img_np = np.array(img)
            
            mean = {
                'CIFAR10': (0.4914, 0.4822, 0.4465),
                'CIFAR100': (0.5071, 0.4867, 0.4408),
                'Imagenette': (0.485, 0.456, 0.406),
            }

            std = {
                'CIFAR10': (0.2023, 0.1994, 0.2010),
                'CIFAR100': (0.2675, 0.2565, 0.2761),
                'Imagenette': (0.229, 0.224, 0.225),
            }
            
            # to torch.Tensor
            tfm_lst = [transforms.ToTensor()]
            if self.dataset != 'FashionMNIST':
                tfm_lst.append(transforms.Normalize(
                    mean[self.dataset], std[self.dataset]
                ))
            if self.dataset == 'Imagenette':
                tfm_lst.insert(0, transforms.CenterCrop(160))
            tfms = transforms.Compose(tfm_lst)
            img_tensor: torch.Tensor = tfms(img).unsqueeze(0)
        else:
            img_tensor = img.unsqueeze(0)
            img_np = np.transpose(img.cpu().numpy(), (1, 2, 0))
            img_np = np.uint8(img_np * 255)
            
        img_tensor = img_tensor.to(self.device)
        
        # get result of classification
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim = 1)
            class_idx, prob = probs.argmax().item(), probs.max().item()
        
        raw_heatmap: torch.Tensor = self._get_raw_heatmap(img_tensor)
        if self.use_relu:
            raw_heatmap = F.relu(raw_heatmap)
            
        raw_heatmap = raw_heatmap.cpu().numpy()
        raw_max, raw_min = raw_heatmap.max(), raw_heatmap.min()
        raw_heatmap = (raw_heatmap - raw_min) / (raw_max - raw_min)
        
        return _get_result(raw_heatmap, img_np, mask_rate), class_idx, prob
    
    @torch.no_grad()
    def _get_feature_maps(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(img_tensor)[self.target_layer]
    
    @abstractmethod
    def _get_raw_heatmap(self, img_tensor: torch.Tensor) -> torch.Tensor:
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
        scores = self.model(img_tensor)
        if use_softmax == 'logits':
            scores = F.softmax(scores, dim = 1)
        idx = scores.argmax(dim = 1)
        scores[:, idx].backward()
        
        handle.remove()
        return grad_outs[0][0].squeeze(0)
    
    def normalize(self, featuremaps: torch.Tensor) -> torch.Tensor:
        feat_reshape = featuremaps.reshape(featuremaps.shape[0], -1)
        maxs = feat_reshape.max(dim = 1).values.reshape(-1, 1, 1)
        mins = feat_reshape.min(dim = 1).values.reshape(-1, 1, 1)
        H = (featuremaps - mins) / (maxs - mins + 1e-5)
        return H
    
    
def _get_result(
        raw_heatmap: np.ndarray, 
        img_np: np.ndarray, 
        mask_rate: float = 0.4
    ) -> np.ndarray:
        
        raw_heatmap = np.uint8(raw_heatmap * 255)
        heatmap = cv2.applyColorMap(
            cv2.resize(raw_heatmap, (img_np.shape[1], img_np.shape[0])),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  
        
        cam_img_np = np.uint8(img_np * (1 - mask_rate) + heatmap * mask_rate)
        return cam_img_np
    
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