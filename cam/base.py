import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 5)
import cv2

from kornia.filters.gaussian import gaussian_blur2d
from sklearn.metrics import auc

from typing import List, Callable, Iterable, Dict, Tuple
from abc import abstractmethod

class_names = {
    'Imagenette': (
        'Tench', 'English Springer', 'Cassette Player', 'Chain Saw', 'Church', 
        'French Horn', 'Garbage Truck', 'Gas Pump', 'Golf Ball', 'Parachute'
    ),
    'CIFAR10': (
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse',
        'Ship', 'Trunk'
    ),
    'FashionMNIST': (
        'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
        'Sneaker', 'Bag', 'Ankle boot'
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
        cuda: int = None
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
        
        if cuda is not None:
            self.device = torch.device(
                f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu'
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
        metric: bool = True,
        saliency: bool = False
    ) -> Tuple[np.ndarray, int, float]:
        # Generate Numpy Image
        metrics = {}
        img_normalized = self.tfm(img.clone().detach().cpu())
        if img_normalized.dim() == 3:
            img_normalized.unsqueeze_(0)
            
        if img.dim() == 4:
            img_np = np.transpose(img.cpu().numpy(), (0, 2, 3, 1))
        elif img.dim() == 3:
            img_np = np.transpose(img.cpu().numpy(), (1, 2, 0))
        else:
            raise ValueError
        img_np = np.uint8(img_np * 255)
                
        saliency_map, pred, prob = self.generate_saliency_map(img_normalized)
        if metric:
            avg_inc, avg_drop = self.calc_avg_inc_drop(
                img_normalized, saliency_map, pred, use_softmax = True
            )
            inse, inse_score = self.calc_causal_metric(
                img_normalized, saliency_map, pred, ins = True
            )
            dele, dele_score = self.calc_causal_metric(
                img_normalized, saliency_map, pred, ins = False
            )
            metrics = {
                'Average Incr': avg_inc, 'Average Drop': avg_drop,
                'Insertion':  inse, 'Deletion': dele,
                'inse_score': inse_score, 'dele_score': dele_score,
            }
            
        heatmaps = []
        for i in range(len(saliency_map)):
            heatmap = cv2.applyColorMap(
                np.uint8(saliency_map[i] * 255),
                cv2.COLORMAP_JET
            )
            heatmaps.append(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        heatmaps = np.asarray(heatmaps)
        cam_np = np.uint8(img_np * (1 - mask_rate) + heatmaps * mask_rate)
        if not saliency:
            return cam_np, pred, prob, metrics
        else:
            return cam_np, saliency_map, pred, prob, metrics 
    
    def calc_avg_inc_drop(
        self, 
        img_normalized: torch.Tensor,
        saliency_map: np.ndarray,
        pred: torch.Tensor,
        use_softmax: bool = True
    ):
        eval_maps = torch.as_tensor(saliency_map).unsqueeze(1) * img_normalized
        with torch.no_grad():
            Y = self._get_scores(img_normalized, use_softmax)
            O = self._get_scores(eval_maps, use_softmax)
        Yc = Y[torch.arange(len(pred)), pred]
        Oc = O[torch.arange(len(pred)), pred]
        
        # Average Increase
        tmp = Yc - Oc
        indices = tmp < 0
        avg_inc = indices.sum() / len(tmp)
        
        # Average Drop
        tmp[tmp < 0] = 0
        avg_drop = (tmp / Yc).mean()

        return avg_inc.item(), avg_drop.item()
    
    def calc_causal_metric(
        self, 
        img_normalized: torch.Tensor,
        saliency_map: np.ndarray,
        pred: torch.Tensor,
        ins: bool = True,
    ):
        tot_pix = np.prod(img_normalized.shape[-2:])
        change_pix = img_normalized.shape[-1] * 2
        n_steps = (tot_pix + change_pix - 1) // change_pix
        salient_order = np.flip(
            np.argsort(saliency_map.reshape(-1, tot_pix), axis = 1), 
            axis = -1
        )
        
        blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))
        if ins:
            start = blur(img_normalized)
            finish = img_normalized.clone()
        else:
            start = img_normalized.clone()
            finish = torch.zeros_like(img_normalized)
        all_scores = np.zeros((n_steps + 1, len(img_normalized)))

        for i in range(n_steps + 1):
            with torch.no_grad():
                scores = self._get_scores(start, use_softmax = True)
            all_scores[i] = scores[torch.arange(len(pred)), pred].cpu().numpy()
            
            if i < n_steps:
                coords = salient_order[:, (change_pix*i):(change_pix*(i + 1))]
                indices = np.arange(len(coords)).reshape(len(coords), 1)
                start.cpu().numpy().reshape(len(img_normalized), -1, tot_pix)[indices, :, coords] = \
                    finish.cpu().numpy().reshape(len(img_normalized), -1, tot_pix)[indices, :, coords]
                
        x_axis = np.linspace(0, 1, n_steps + 1)
        metrics = [auc(x_axis, all_scores[:, i]) for i in range(len(img_normalized))]
        return np.mean(metrics), all_scores.mean(axis = 1)

    @torch.no_grad()
    def model_predict(
        self, 
        img_normalized: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:            
        self.model.to(self.device)
        probs = F.softmax(self.model(img_normalized.to(self.device)), dim = 1)
        max_info = probs.max(dim = 1)
        return max_info.indices.cpu(), max_info.values.cpu()
    
    def hook_featuremap(self):
        def hook_fn(_, __, output):
            self.features.append(output)
        return hook_fn
    
    def generate_saliency_map(
        self, 
        img_normalized: torch.Tensor,
    ) -> Tuple[np.ndarray, int, float]:        
        # extract_features
        self.features = []
        target_layer = dict([*self.model.named_children()])[self.target_layer]
        handle = target_layer.register_forward_hook(self.hook_featuremap())
        pred, prob = self.model_predict(img_normalized)
        handle.remove()
        self.featuremaps: torch.Tensor = self.features[0]
        
        raw_saliency_map: torch.Tensor = self._get_raw_saliency_map(img_normalized, pred)
        if self.use_relu:
            raw_saliency_map = F.relu(raw_saliency_map)
        raw_max = torch.max(
            raw_saliency_map.reshape(len(img_normalized), -1), 
            dim = 1
        ).values.reshape(-1, 1, 1)
        raw_min = torch.min(
            raw_saliency_map.reshape(len(img_normalized), -1), 
            dim = 1
        ).values.reshape(-1, 1, 1)
        raw_saliency_map = (raw_saliency_map - raw_min) / (raw_max - raw_min)
        saliency_maps: torch.Tensor = transforms.Resize(img_normalized.shape[-1])(raw_saliency_map)
        # saliency_maps = saliency_maps.nan_to_num(0.0)
        
        return saliency_maps.cpu().numpy(), pred, prob
    
    @abstractmethod
    def _get_raw_saliency_map(
        self, 
        img_normalized: torch.Tensor,
        pred: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
    
    def _get_layer_names(self) -> List[str]:
        layer_names = []
        for name, _ in self.model.named_children():
            layer_names.append(name)
        return layer_names
    
    def _get_scores(self, img_normalized: torch.Tensor, use_softmax = True):
        self.model.to(self.device)
        scores = self.model(img_normalized.to(self.device))
        if use_softmax:
            scores = F.softmax(scores, dim = 1)
        return scores
    
    def _get_grads(self, img_normalized: torch.Tensor, pred: torch.Tensor, use_softmax: bool):
        grad_outs = []
        handle = self.model.get_submodule(self.target_layer).register_full_backward_hook(
            lambda _, __, go: grad_outs.append(go)
        )
        
        self.model.zero_grad()
        scores = self._get_scores(img_normalized, use_softmax)
        gathered_scores = scores.gather(1, pred.to(self.device).reshape(-1, 1)).sum()
        gathered_scores.backward()
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
        
def plot_casual_metrics(
    scores: np.ndarray,
    cam_name: str,
    ins: bool = True,
    save_pth: str = None
):  
    fontsize = 18
    tick_fontsize = 15
    
    x_axis = np.linspace(0, 1, len(scores))
    plt.plot(x_axis, scores)
    plt.fill_between(x_axis, 0, scores, alpha=0.4)
    if ins:
        mode = 'Insertion' 
        xlabel = 'Pixel Inserted'
    else:
        mode = 'Deletion'
        xlabel = 'Pixel Deleted'
    fig_name = f'{cam_name} {mode} Curve'
    plt.title(fig_name, fontsize = fontsize)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.tick_params(axis='both',labelsize = tick_fontsize)
    plt.xlim([0.0, 1.00])
    plt.ylim([0.0, 1.02])
    
    if save_pth is not None:
        plt.savefig(save_pth + fig_name + 'png')
        plt.savefig(save_pth + fig_name + 'pdf')
    else:
        plt.show()