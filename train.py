import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import timm

from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import logging
import time
import argparse
from tabulate import tabulate

import utils
import trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic Settings')
    parser.add_argument('--model_mode', default = 'resnet18')
    parser.add_argument('--dataset', default = 'FashionMNIST')
    parser.add_argument('--bs', default = 128, type = int)
    parser.add_argument('--lr', default = 0.01, type = float)
    parser.add_argument('--epochs', default = 25, type = int)
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--cuda', default = 0, type = int)
    parser.add_argument('--use_lr_sche', action = 'store_true')
    parser.add_argument('--use_wandb', action = 'store_true')
    args = parser.parse_args()
    print(tabulate(
        list(vars(args).items()), headers = ['attr', 'setting'], tablefmt ='orgtbl'
    ))
    
    trainer_model = trainer.Trainer(**vars(args))
    trainer_model.fit(args.epochs)