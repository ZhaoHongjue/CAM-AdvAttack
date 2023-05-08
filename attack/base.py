import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

import numpy as np
from tqdm import trange

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

class BaseAttack:
    def __init__(
        self,
        model: nn.Module,
        dataset: str,
        cuda: int = None
    ) -> None:
        self.model = model
        self.tfm = transforms.Normalize(mean[dataset], std[dataset])
        if cuda is not None:
            self.device = torch.device(
                f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        
    def __call__(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor = None,
        max_iter: int = None,
        num_classes: int = None,
        attack_kwargs: dict = None
    ) -> torch.Tensor:
        pass
        
    def attack_one(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError