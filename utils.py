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
from typing import Iterable, Dict, Callable

import cam

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