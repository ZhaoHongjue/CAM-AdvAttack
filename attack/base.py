import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from tqdm import trange

class BaseAttack:
    def __init__(
        self,
        model: nn.Module,
        cuda: int = None
    ) -> None:
        self.model = model
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