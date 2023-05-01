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
