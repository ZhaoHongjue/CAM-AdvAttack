import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
from tabulate import tabulate
import argparse

from trainer import Trainer, generate_data_iter
import utils
import cam

def test_single_cam(cam_method, model_mode, dataset, cuda = 0, seed = 0, reload = False):
    if model_mode == 'resnet18': target_layer = 'layer4'
    elif model_mode == 'efficientnet_b0': target_layer = 'conv_head'
    elif model_mode == 'densenet121': target_layer = 'features'
    else: target_layer = 'blocks'
    fc_layer = 'fc' if model_mode == 'resnet18' else 'classifier'
    
    if cam_method == 'CAM':
        assert model_mode == 'resnet18' or 'densenet121'
    fig_num = 100

    utils.set_random_seed(seed)
    trainer = Trainer(
        model_mode, dataset = dataset, bs = 128,
        lr = 0.01, seed = seed, cuda = cuda,
        use_lr_sche = True, use_wandb = False,
    )
    trainer.load()
    
    test_iter = generate_data_iter(dataset, batch_size = fig_num, mode = 'test')
    imgs, labels = next(iter(test_iter))
    
    df_pth = f'./thesis/cam_metrics/{dataset}/'
    if not os.path.exists(df_pth):
        os.makedirs(df_pth)
        
    cam_np_pth = f'thesis/cam_pics/{dataset}/'
    if not os.path.exists(cam_np_pth):
        os.makedirs(cam_np_pth)
        
    mycam = eval(f'cam.{cam_method}')(trainer.model, dataset, target_layer, fc_layer, cuda = cuda)
    cam_imgs, _, __, metrics_cam = mycam(imgs, metric = True)
    metrics = {
        'Average Incr': metrics_cam['Average Incr'], 'Average Drop': metrics_cam['Average Drop'],
        'Insertion':  metrics_cam['Insertion'], 'Deletion': metrics_cam['Deletion'],
    }
    np.save(cam_np_pth + f'{cam_method}-{dataset}-{model_mode}-seed{seed}.npy', cam_imgs)
    print(tabulate(
        list(metrics.items()), tablefmt ='orgtbl'
    ))
    
    metric_df_pth = df_pth + f'CAM-{model_mode}-{dataset}-seed{seed}.csv'
    if not os.path.exists(metric_df_pth) or reload:
        metrics_df = pd.DataFrame(list(metrics.values()), columns = [cam_method], index = list(metrics.keys()))
    else:
        metrics_df = pd.read_csv(metric_df_pth, index_col = 0)
        metrics_df[cam_method] = metrics
    metrics_df.to_csv(metric_df_pth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Basic Settings')
    parser.add_argument('--method', default = 'ScoreCAM')
    parser.add_argument('--model_mode', default = 'resnet18')
    parser.add_argument('--dataset', default = 'FashionMNIST')
    parser.add_argument('--cuda', default = 0, type = int)
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--reload', action = 'store_true')
    args = parser.parse_args()
    print(tabulate(
        list(vars(args).items()), headers = ['attr', 'setting'], tablefmt ='orgtbl'
    ))
    
    cams = [
        'CAM', 'GradCAM', 'GradCAMpp', 'SMGradCAMpp', #'LayerCAM', 'XGradCAM', 'ScoreCAM',  'SSCAM', 'ISCAM'
    ]
    if args.method != 'all':
        assert args.method in cams
        print('######################################')
        test_single_cam(args.method, args.model_mode, args.dataset, args.cuda, args.seed, args.reload)
    else:
        print('######################################')
        for method in cams:
            print(method)
            test_single_cam(method, args.model_mode, args.dataset, args.cuda, args.seed)
            print('######################################')