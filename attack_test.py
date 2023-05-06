import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time
import os
from tabulate import tabulate
import argparse
from typing import List

from trainer import Trainer, generate_data_iter
import utils
import attack
import cam

num_classes = 10
settings = {
    'FashionMNIST': {
        'FGSM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.03},
        },
        'FGM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.8},
        },
        'StepLL': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.035},
        },
        'IFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.015},
        },
        'MIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.015},
        },
        'NIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.015},
        },
        'PGD': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.035},
        },
        'IterLL': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.012},
        },
        'DeepFool': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {},
        },
        'LBFGS': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'c': 0.05},
        },
    },
    
    'CIFAR10': {
        'FGSM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'FGM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 1},
        },
        'StepLL': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'IFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.01},
        },
        'MIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.01},
        },
        'NIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.01},
        },
        'PGD': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'IterLL': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.012},
        },
        'DeepFool': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {},
        },
        'LBFGS': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'c': 0.06},
        },
    },
    
    'Imagenette': {
        'FGSM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'FGM': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 5},
        },
        'StepLL': {
            'max_iter': None,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'IFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.01},
        },
        'MIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.01},
        },
        'NIFGSM': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.01},
        },
        'PGD': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.02},
        },
        'IterLL': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'eps': 0.012},
        },
        'DeepFool': {
            'max_iter': 100,
            'num_classes': num_classes,
            'attack_kwargs': {},
        },
        'LBFGS': {
            'max_iter': 10,
            'num_classes': num_classes,
            'attack_kwargs': {'c': 0.3},
        },
    },
}

def test_single_advatt(
    advatt: attack.BaseAttack, 
    model_mode: str,
    dataset: str,
    scorecam: cam.ScoreCAM,
    suc_imgs: torch.Tensor,
    suc_labels: torch.Tensor,
    pths: List[str],
    cuda = 0, 
    seed = 0, 
    reload = False
):
    metric_pth, attack_pth, indices_pth = pths
    att_name = advatt.__class__.__name__
    start = time.time()
    att_imgs = advatt(
        suc_imgs, suc_labels, **settings[dataset][att_name]
    )
    finish = time.time()
    np.save(attack_pth + f'{att_name}-{dataset}-{model_mode}-seed{seed}.npy', att_imgs.numpy())
    deltas = att_imgs - suc_imgs
    att_preds, _ = scorecam.model_predict(scorecam.tfm(att_imgs))
    
    # Success Rate
    att_indices = att_preds != suc_labels
    np.save(indices_pth + f'{att_name}-{dataset}-{model_mode}.npy', att_indices.cpu().numpy())
    success_rate = (att_indices.sum() / len(att_preds)).item()
    
    # delta norm
    delta_norm = torch.mean(
        torch.linalg.norm(deltas.reshape(len(deltas), -1), dim = 1) \
            / torch.linalg.norm(suc_imgs.reshape(len(deltas), -1), dim = 1)
    ).item()
    
    # In cam part, we only focus on successful part
    att_suc_cams, att_suc_saliency, _, __, att_suc_cam_metrics \
        = scorecam(att_imgs[att_indices], metric = True, saliency = True)
    
    # delta saliency map norm
    saliency_diff = np.linalg.norm(
        att_suc_saliency - suc_saliency_maps[att_indices.numpy()],
        axis = (1, 2)
    ).mean()
    
    # maximum shift
    size = suc_imgs.shape[-1]
    suc_max_idx_raw = np.argmax(suc_saliency_maps[att_indices.numpy()].reshape(att_indices.sum(), -1), axis = 1)
    suc_max_x, suc_max_y = suc_max_idx_raw // size, suc_max_idx_raw % size

    att_max_idx_raw = np.argmax(att_suc_saliency.reshape(len(att_suc_saliency), -1), axis = 1)
    att_max_x, att_max_y = att_max_idx_raw // size, suc_max_idx_raw % size
    shift_dist = np.sqrt((att_max_x - suc_max_x)**2 + (att_max_y - suc_max_y)**2).mean()
    
    metrics = {
        'time': finish - start,
        'success_rate': success_rate,
        'delta_norm': delta_norm,
        'Average Incr': att_suc_cam_metrics['Average Incr'], 
        'Average Drop': att_suc_cam_metrics['Average Drop'],
        'Insertion':  att_suc_cam_metrics['Insertion'], 
        'Deletion': att_suc_cam_metrics['Deletion'],
        'saliency_diff': saliency_diff,
        'shift_dist': shift_dist
    }
    print(tabulate(
        list(metrics.items()), tablefmt ='orgtbl'
    ))
    
    metric_df_pth = metric_pth + f'Attack-{model_mode}-{dataset}-seed{seed}.csv'
    if not os.path.exists(metric_df_pth):
        metrics_df = pd.DataFrame(
            list(metrics.values()), 
            columns = [att_name], 
            index = list(metrics.keys())
        )
    else:
        metrics_df = pd.read_csv(metric_df_pth, index_col = 0)
        metrics_df[att_name] = metrics
    metrics_df.to_csv(metric_df_pth)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic Settings')
    parser.add_argument('--method', default = 'FGSM')
    parser.add_argument('--model_mode', default = 'resnet18')
    parser.add_argument('--dataset', default = 'FashionMNIST')
    parser.add_argument('--cuda', default = 0, type = int)
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--reload', action = 'store_true')
    args = parser.parse_args()
    print(tabulate(
        list(vars(args).items()), headers = ['attr', 'setting'], tablefmt ='orgtbl'
    ))
    
    # Make Dirs
    metric_pth = f'./thesis/attack/attack_metrics/{args.dataset}/'
    if not os.path.exists(metric_pth):
        os.makedirs(metric_pth)
        
    attack_pth = f'./thesis/attack/attack_pics/{args.dataset}/'
    if not os.path.exists(attack_pth):
        os.makedirs(attack_pth)
        
    indices_pth = f'./thesis/attack/indices/{args.dataset}/'
    if not os.path.exists(indices_pth):
        os.makedirs(indices_pth)
    
    ## Model and test data load
    fig_num = 100
    utils.set_random_seed(args.seed)
    trainer = Trainer(
        args.model_mode, dataset = args.dataset, bs = 128,
        lr = 0.01, seed = args.seed, cuda = args.cuda,
        use_lr_sche = True, use_wandb = False,
    )
    trainer.load()
    test_iter = generate_data_iter(args.dataset, batch_size = fig_num, mode = 'test')
    imgs, labels = next(iter(test_iter))
    
    ## create cam
    if args.model_mode == 'resnet18': target_layer = 'layer4'
    elif args.model_mode == 'efficientnet_b0': target_layer = 'conv_head'
    elif args.model_mode == 'densenet121': target_layer = 'features'
    else: target_layer = 'blocks'
    scorecam = cam.ScoreCAM(
        trainer.model, args.dataset, target_layer, cuda = args.cuda
    )
    
    raw_preds, _ = scorecam.model_predict(scorecam.tfm(imgs))
    suc_indices = raw_preds == labels
    np.save(indices_pth + f'suc-{args.dataset}-{args.model_mode}.npy', suc_indices.cpu().numpy())
    suc_imgs, suc_labels = imgs[suc_indices], labels[suc_indices]
    suc_cams, suc_saliency_maps, suc_preds, suc_probs, _ \
        = scorecam(suc_imgs, metric = False, saliency = True)
    
    pths = [metric_pth, attack_pth, indices_pth]
    if args.method != 'all':
        advatt = eval(f'attack.{args.method}')(trainer.model, args.cuda)
        test_single_advatt(
            advatt, args.model_mode, args.dataset, scorecam, suc_imgs, suc_labels, pths, args.cuda, args.seed
        )
    else:
        attacks = [
            'FGSM', 'FGM', 'StepLL', 
            # 'IFGSM', 'MIFGSM', 'NIFGSM', 'IterLL', 'PGD',
            # 'DeepFool', 'LBFGS'
        ]
        for att in attacks:
            print('######################################################')
            print(att)
            advatt = eval(f'attack.{att}')(trainer.model, args.cuda)
            test_single_advatt(
                advatt, args.model_mode, args.dataset, scorecam, suc_imgs, suc_labels, pths, args.cuda, args.seed
            )
    
    
    
    
