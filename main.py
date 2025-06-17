import sys
sys.path.append('..')

# OS and file handling
import os
import gc
import argparse
from argparse import Namespace
import pickle
import json

# utils
from functools import partial
from copy import deepcopy
from tqdm import tqdm
import time

# mathematics
import numpy as np
import random

# data handling
import pandas as pd

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable

# Torchvision
import torchvision

# logging
# import wandb

# my_lib
from model.net import *
from model.train import train, test, NoiseInjector
from misc.utils import save_checkpoint, load_transform, load_dataset


def main(args0):
    args = deepcopy(args0)

    # seed reset
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set save path
    print('Device:', args.device, 'Seed:', args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    # load dataset and transform
    transforms = load_transform(args)
    train_dataset, valid_dataset, test_dataset = load_dataset(transforms, args)
    if args.noise_rate > 0:
        print('Corrupting the training set with label noise rate:', args.noise_rate)
    print(f"Train dataset size: {len(train_dataset)}, Valid dataset size: {len(valid_dataset)}, Test dataset size: {len(test_dataset)}")

    # dataloader
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f'torch train, valod, test loader loaded with Batch size {args.batch_size}')

    curr_net, best_net = load_model(args)
    if args.batch_size >= len(train_dataset):
        args.batch_size = len(train_dataset) # full-batch gradient descent (GD)
        args.optim = 'gd'
    optimizer = optim.SGD(curr_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = None
    if args.optim_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.tot_epoch)
        print('Cosine annealing scheduler initialized.')

    print('Perturbation type:', args.perturb_type, 'Perturbation eps:', args.perturb_eps, 'Perturbation tau:', args.perturb_tau)
    print('Model:', args.model, 'Optimizer:', args.optim, 'Scheduler:', args.optim_scheduler, 'Learning rate:', args.lr, 'Weight decay:', args.weight_decay, 'Momentum:', args.momentum)

    # noise injector
    noise_injector = None
    noise_injector_flag = False
    if args.perturb_type != 'none' and args.perturb_eps > 0:
        print('Noise injector initialized.')
        noise_injector = NoiseInjector(curr_net, args)
        noise_injector_flag = True

    # train the model
    print('Start training... until epoch:', args.tot_epoch)
    best_metric = np.inf if args.best_metric == 'loss' else 0.0
    is_best = False

    ret_train = {}
    ret_test = {}
    for epoch in range(1, args.tot_epoch + 1):
        if epoch > args.tot_epoch * 0.8 and noise_injector_flag:
            print('Remove noise injector for the convergence.')
            noise_injector = None # remove noise injector after 90% of training
            noise_injector_flag = False

        train_log = train(curr_net, trainloader, optimizer, args, noise_injector=noise_injector, save_acc = True, epoch=epoch)
        test_log = test(curr_net, testloader, args, epoch=epoch)
        if scheduler is not None:
            scheduler.step()
        print(f'Epoch: {epoch}, Train_loss(avg.): {np.mean(train_log['loss'][0]):.4f}, Train_acc(avg.): {np.mean(train_log['acc'][0].item()):.2f}, Test_loss: {test_log['loss'][0]:.4f}, Test_acc: {test_log['acc'][0].item():.2f}')

        # save logs
        for key in train_log.keys():
            if key not in ret_train.keys():
                ret_train[key] = []
            ret_train[key] += train_log[key]
        for key in test_log.keys():
            if key not in ret_test.keys():
                ret_test[key] = []
            ret_test[key] += test_log[key]

        # save the best model
        is_best = test_log['loss'][0] < best_metric if args.best_metric == 'loss' else test_log['acc'][0].item() > best_metric
        if is_best:
            best_metric = min(test_log['loss'][0], best_metric) if args.best_metric == 'loss' else max(test_log['acc'][0].item(), best_metric)
            best_net.load_state_dict(curr_net.state_dict())
            print(f'Best model updated. {args.best_metric}: {best_metric:.4f}')

        # save the checkpoint
        if epoch % 50 == 0:
            save_checkpoint(epoch, curr_net, optimizer, best_net, args.save_path)
            print('Checkpoint saved.')

    # save the final model
    save_checkpoint(epoch, curr_net, optimizer, best_net, args.save_path)
    print('Final model saved.')

    # test the best model
    best_test_log = test(best_net, testloader, args)
    print(f'Best model test loss: {best_test_log['loss'][0]:.4f}, test acc: {best_test_log['acc'][0].item():.2f}')

    # save logs and hyperparameters
    train_df = pd.DataFrame(ret_train)
    test_df = pd.DataFrame(ret_test)

    train_df.to_csv(os.path.join(args.save_path, "train_log.csv"), index=False)
    test_df.to_csv(os.path.join(args.save_path, "test_log.csv"), index=False)

    # save results as .pkl with pickles
    hparams = json.dumps(vars(args))
    with open(os.path.join(args.save_path, "hparams.json"), "w") as f:
        f.write(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SGD"
    )
    parser.add_argument(
        "--save-path",
        default="./checkpoint",
        type=str,
        metavar="PATH",
        help="path to save result (default: none)",
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='cifair', 
        help="choice for dataset (default: cifair)"
    )
    parser.add_argument(
        '--path-data', 
        type=str, 
        default='./data', 
        help="data path (default: ./data)"
    )
    parser.add_argument(
        '--split', 
        type=int, 
        default=0, 
        help="dataset split for small datsets"
    )
    parser.add_argument(
        '--noise-rate', 
        type=float, 
        default=0, 
        help="add noise to labels with the amount of noise-rate"
    )
    parser.add_argument(
        '--noisy-valid', 
        type=int, 
        default=0, 
        help="whether to currpt validation set or not"
    )
    parser.add_argument(
        '--class-dependent', 
        type=int, 
        default=0, 
        help="whether label noise is class dependent or not"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='resnet18', 
        help="choice for model (default: resnet18)"
    )
    parser.add_argument(
        '--best-metric', 
        type=str, 
        default='loss', 
        help="which metric to track best model (default: loss, choose between loss or acc)"
    )
    parser.add_argument(
        '--tot-epoch', 
        type=int, 
        default=100
    )
    parser.add_argument(
        '--optim', 
        type=str, 
        default='sgd', 
        help="choice for optimizer (default: sgd)"
    )
    parser.add_argument(
        '--optim-scheduler', 
        type=str, 
        default='none', 
        help="choice for optimizer scheduler (default: none)"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-2, 
        help="learning rate for optimizer (default: 1e-2)"
    )
    parser.add_argument(
        '--momentum', 
        type=float, 
        default=0.0, 
        help="momentum for optimizer (default: 0.0)"
    )
    parser.add_argument(
        '--weight-decay', 
        type=float, 
        default=0.0, 
        help="weight decay for optimizer (default: 0.0)"
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32, 
        help="batch size (default: 32)"
    )
    parser.add_argument(
        '--test-batch-size', 
        type=int, 
        default=4096
    )
    parser.add_argument(
        '--valid-ratio', 
        type=float, 
        default=0.2
    )
    parser.add_argument(
        '--num-workers', 
        type=int, 
        default=4, 
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1
    )
    parser.add_argument(
        '--loss-type', 
        type=str, 
        default='ce', 
        help="loss type; currently ce implemented"
    )
    parser.add_argument(
        '--perturb-type', 
        type=str, 
        default='none', 
        help="perturb type; currently only random and anti implemented (default: none)"
    )
    parser.add_argument(
        '--perturb-eps0', 
        type=float, 
        default=0, 
        help="(anti-noise) how much to perturb the network via white noise (default: 0.0)"
    )
    parser.add_argument(
        '--perturb-eps', 
        type=float, 
        default=0, 
        help="how much to perturb the network (via colored noise for the anti noise) (default: 0)"
    )
    parser.add_argument(
        '--perturb-tau', 
        type=float, 
        default=0, 
        help="(anti-noise) tau for anti noise (default: 0)"
    )
    
    args = parser.parse_args()
    if args.data == 'cifar10':
        args.num_classes = 10
        args.trainset_size = 50000
        args.validset_size = 10000
    elif args.data == 'cifar100':
        args.num_classes = 100
        args.trainset_size = 50000
        args.validset_size = 10000
    elif args.data == 'cifair':
        args.num_classes = 10
        args.trainset_size = 500
        args.validset_size = 10000
        
    # args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    main(args)
    