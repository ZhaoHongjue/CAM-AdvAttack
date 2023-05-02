import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data

import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST

import numpy as np
import matplotlib.pyplot as plt

import os
import yaml
from typing import Iterable, Dict, Callable, Tuple

import cam
import attack

classes = {
    'imagenette': (
        'tench', 'English springer', 'cassette player', 'chain saw', 'church', 
        'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute'
    ),
    'CIFAR10': (
        'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse',
        'Ship', 'Trunk'
    )
}

# ===========================================================
#                    Related Classes
# ===========================================================

class Accumulator:
    '''used to accumulate related metrics'''
    def __init__(self, n: int):
        self.arr = [0] * n
        
    def add(self, *args):
        self.arr = [a + float(b) for a, b in zip(self.arr, args)]
    
    def __getitem__(self, idx: int):
        return self.arr[idx]


class VerboseExe:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles = []
        
        for name, module in self.model.named_children():
            module.__name__ = name
            self.handles.append(
                module.register_forward_hook(
                    lambda m, _, o: print(
                        f'{m.__name__:10}\t\t\t{o.shape}'
                    ) 
                )
            )
                
    def __call__(self, X) -> None:
        self.model(X)
        for i in range(len(self.handles)):
            self.handles[i].remove()    
    
    
# ===========================================================
#                    Related Functions
# ===========================================================

def set_random_seed(seed: int):    
    '''
    set random seed
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

@torch.no_grad()
def model_predict(
    model: nn.Module, 
    batch_imgs: torch.Tensor,
    device: str or torch.device,
    normalize_mode: str = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    predict results
    '''
    if normalize_mode is not None:
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
        tfm = transforms.Normalize(
            mean[normalize_mode], std[normalize_mode]
        )
        imgs_clone = tfm(batch_imgs.clone())
    else:
        imgs_clone = batch_imgs.clone()
        
    if type(device) == str:
        device = torch.device(device)
    model.to(device)
    probs = F.softmax(model(imgs_clone.to(device)), dim = 1)
    max_info = probs.max(dim = 1)
    return max_info.indices.cpu(), max_info.values.cpu()

def evaluate_attack(
    labels: torch.Tensor,
    raw_pred: torch.Tensor,
    attacked_pred: torch.Tensor
):
    indices = labels == raw_pred
    raw_acc = indices.sum() / len(labels)
    
    success_num = (attacked_pred[indices] != raw_pred[indices]).sum() 
    success_rate = success_num / indices.sum()
    
    return raw_acc, success_rate
    
def plot_img(
    dataset: str,
    attack_method: str,
    model: nn.Module, 
    img: torch.Tensor, 
    attack_img: torch.Tensor,
    save: bool = False
):
    plt.clf()
    fig = plt.figure(figsize = (9, 9))
    plt.subplots_adjust(wspace = 0.04, hspace = 0.04)
    
    mycam = cam.CAM(model, 'layer4', 'fc')
    raw_cam_img, raw_idx, raw_prob = mycam(img.cpu())
    attack_cam_img, attack_idx, attack_prob = mycam(attack_img.cpu())
    
    plt.subplot(2, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Before Attack', fontsize = 20)
    plt.ylabel('Raw Image', fontsize = 20)
    plt.imshow(np.transpose(np.uint8(img.cpu().numpy()*255), (1, 2, 0)))

    plt.subplot(2, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('After Attack', fontsize = 20)
    plt.imshow(np.transpose(np.uint8(attack_img.cpu().numpy()*255), (1, 2, 0)))
    
    plt.subplot(2, 2, 3)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f'{classes[dataset][raw_idx]} ({raw_prob*100:4.2f}%)', fontsize = 20)
    plt.ylabel('CAM', fontsize = 20)
    plt.imshow(raw_cam_img)
    
    plt.subplot(2, 2, 4)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f'{classes[dataset][attack_idx]} ({attack_prob*100:4.2f}%)', fontsize = 20)
    plt.imshow(attack_cam_img)
    
    pth = f'./attack_test_pic/{attack_method}/'
    if not os.path.exists(pth):
        os.makedirs(pth)
    
    if save:
        plt.savefig(pth + f'{dataset}.pdf', bbox_inches = 'tight')
        plt.savefig(pth + f'{dataset}.png', bbox_inches = 'tight')
    plt.show()