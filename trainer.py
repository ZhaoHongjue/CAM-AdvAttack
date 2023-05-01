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
import wandb

import utils

class Trainer:
    def __init__(
        self, model_mode: str, dataset: str, bs: int = 128, lr: float = 0.1, 
        seed: int = 0, cuda: int = 0, use_lr_sche: bool = True, 
        use_wandb: bool = True, **kwargs
    ) -> None:
        self. model_mode, self.dataset, self.seed = model_mode, dataset, seed
        # Create model
        self.model = create_model(model_mode, dataset)
        self.model_name = f'{model_mode}-{dataset}-bs{bs}-lr{lr}-seed{seed}'
        self.makedirs()
        
        # Basic Components
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(
            self.model.parameters(), lr = lr, momentum = 0.9, 
            nesterov = True
        )
        if use_lr_sche:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, 'max', factor = 0.5, patience = 10
            )
        self.device = torch.device(
            f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu'
        )
        
        self.train_iter = generate_data_iter(dataset, bs, train = True)
        self.val_iter = generate_data_iter(dataset, bs, train = False)

        logging.basicConfig(
            filename = self.log_pth + self.model_name + '.log', 
            level = logging.INFO
        )
        
        self.use_wandb = False
    
    def fit(self, epochs: int = 100):
        utils.set_random_seed(self.seed)
        metrics = {
            'train_loss': [], 'test_loss': [],
            'train_acc': [], 'test_acc': []
        }
        max_acc, best_epoch = 0.0, 0
        
        logging.info(f'train on {self.device}')
        self.model.to(self.device)
        
        for epoch in range(epochs):
            start = time.time()
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            test_loss, test_acc = self.val_epoch()
            if hasattr(self, 'scheduler'):
                self.scheduler.step(test_acc)
            final = time.time()
            
            if test_acc > max_acc:
                max_acc = test_acc
                best_epoch = epoch
                torch.save(
                    self.model.state_dict(), 
                    self.model_pth + self.model_name
                )
                
            for metric in list(metrics.keys()):
                metrics[metric].append(eval(metric))
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                })
            
            if epoch % 10 == 0:
                pd.DataFrame(metrics).to_csv(
                    self.metric_pth + self.model_name + '.csv'
                )
                
            train_info = f'train loss: {train_loss:.2e},  train acc: {train_acc * 100:4.2f}%'
            test_info = f'test loss: {test_loss:.2e},  test acc: {test_acc * 100:4.2f}%'
            other_info = f'time: {final-start:.2f},  best epoch: {best_epoch}'
            info = f'epoch: {epoch:3},  {train_info},  {test_info},  {other_info}'
            logging.info(info)
            
            print(info)
    
    def train_epoch(self, epoch, epochs):
        self.model.train()
        accu = utils.Accumulator(3)
        with tqdm(enumerate(self.train_iter), total = len(self.train_iter), leave = True) as t:
            for idx, (X, Y) in t:
                self.opt.zero_grad()
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat: torch.Tensor = self.model(X)
                loss: torch.Tensor = self.loss_fn(Y_hat, Y)
                loss.backward()
                self.opt.step()
                with torch.no_grad():
                    correct_num = get_correct_num(Y, Y_hat)
                accu.add(loss.item() * len(Y), correct_num, len(Y))
                t.set_description(f'Epoch: [{epoch}/{epochs}]')
                t.set_postfix({
                    'batch': f'{idx} / {len(self.train_iter)}',
                    'training loss': f'{accu[0] / accu[-1]:.2e}',
                    'training acc': f'{(accu[1] / accu[-1]) * 100:4.2f}%'
                })
        return accu[0] / accu[-1], accu[1] / accu[-1]

    def val_epoch(self):
        self.model.eval()
        accu = utils.Accumulator(3)
        with torch.no_grad():
            for X, Y in self.val_iter:
                X, Y = X.to(self.device), Y.to(self.device)
                Y_hat: torch.Tensor = self.model(X)
                loss: torch.Tensor = self.loss_fn(Y_hat, Y)
                correct_num = get_correct_num(Y, Y_hat)
                accu.add(loss.item() * len(Y), correct_num, len(Y))
        return accu[0] / accu[-1], accu[1] / accu[-1]
    
    def load(self):
        load_pth = self.model_pth + self.model_name
        self.model.load_state_dict(torch.load(load_pth, map_location = 'cpu'))
        self.model.eval()  
        
    def makedirs(self):
        self.results_pth = f'./results/{self.dataset}/{self.model_mode}'
        self.model_pth = f'{self.results_pth}/models/'
        self.metric_pth = f'{self.results_pth}/metrics/'
        self.log_pth = f'{self.results_pth}/logs/'
        
        pths = [self.model_pth, self.metric_pth, self.log_pth]
        for pth in pths:
            if not os.path.exists(pth):
                os.makedirs(pth)
    
def create_model(model_mode: str, dataset: str) -> nn.Module:
    '''
    Create model based on timm
    '''
    model: nn.Module = timm.create_model(model_mode, pretrained = True, num_classes = 10)
    if model_mode == 'resnet18':
        if dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'FashionMNIST':
            in_channels = 1 if dataset == 'FashionMNIST' else 3
            model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size = 3, stride = 1, padding = 1
            )
            model.maxpool = nn.Identity()        
    return model

def generate_data_iter(dataset: str, batch_size: int = 128, train: bool = True):
    '''
    Generate data iterator
    '''
    data_pth = f'./data/{dataset}/'
    if not os.path.exists(data_pth):
        os.makedirs(data_pth)
        
    if dataset == 'CIFAR10' or dataset == 'CIFAR100' or dataset == 'FashionMNIST':
        if dataset == 'FashionMNIST': tfm = transforms.ToTensor()
        else:
            tfm = transforms.Compose([
                transforms.AutoAugment(),
                transforms.ToTensor(),
            ])
        return data.DataLoader(
            eval(dataset)(
                root = data_pth, train = train, download = True, transform = tfm
            ),
            batch_size = batch_size, shuffle = train
        )
    else:
        pass

def get_correct_num(Y: torch.Tensor, Y_hat: torch.Tensor):
    with torch.no_grad():
        if Y_hat.dim() > 1 and Y_hat.shape[1] > 1:
            Y_hat = F.softmax(Y_hat, dim = 1)
            Y_hat = Y_hat.argmax(dim = 1)
        cmp = Y_hat.type(Y.dtype) == Y
        return float(cmp.type(Y.dtype).sum())