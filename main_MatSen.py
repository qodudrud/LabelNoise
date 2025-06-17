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

# Torchvision
import torchvision

# my_lib
from model.train_matsens import MatrixSensingModel, train, validate


def main(args0):
    args = deepcopy(args0)

    # seed for genearing the low-rank matrix and the training data
    # torch.backends.cudnn.benchmark = True  # Enable benchmark mode for faster training (but may not be reproducible)
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility (but may be slower for some Conv operations)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # set save path 
    print('Device:', args.device, 'Seed:', args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    # Generate low-rank matrix X_star = U U^T
    U_true = torch.randn(args.n_dim, args.rank, device=args.device)
    X_star = U_true @ U_true.T  # [n, n]
    X_star /= torch.norm(X_star, p='fro')  # Normalize to unit Frobenius norm

    eigvs = torch.linalg.eigvals(X_star)
    print('Eigenvalues of X_star:', eigvs)

    # Generate random sensing matrices A_i and labels y_i = Tr(A_i^T X_star)
    A_true = torch.randn(args.m_samples, args.n_dim, args.n_dim, device=args.device)  # [m, n, n]
    y_true = torch.einsum('mij,ij->m', A_true, X_star)  # [m]

    # Split into train/test
    if args.m_train > args.m_samples:
        args.m_train = args.m_samples  # Ensure m_train is not greater than m_samples
    A_train, y_train = A_true[:args.m_train], y_true[:args.m_train]
    A_test, y_test = A_true[args.m_train:], y_true[args.m_train:]

    # Add noise to the training labels if specified
    y_train += torch.randn_like(y_train, device=args.device) * args.noise_rate

    # seed reset for the training process
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('Training with m_train:', args.m_train, 'm_samples:', args.m_samples, 'n_dim:', args.n_dim, 'rank:', args.rank, 
          'noise_rate:', args.noise_rate)
    print('Train set size:', A_train.shape, 'Test set size:', A_test.shape)

    args.sigma_rho = args.sigma * np.sqrt(1 - (args.rho ** 2))
    print('injecting noise:', args.inject_noise, 'sigma:', args.sigma, 'rho:', args.rho, 'sigma_rho:', args.sigma_rho)
    
    # model initialization
    model = MatrixSensingModel(args).to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')
    print("Parameters for optimizer, batch size:", args.batch_size, "learning rate:", args.lr)

    # train the model using gradient descent
    print('Start training... until iter:', args.tot_iter)
    res_dict = train(model, optimizer, criterion, A_train, y_train, A_test, y_test, args)
    
    print('Training completed.')
    print('Final train loss:', res_dict['train_loss'][-1])
    print('Final test loss:', res_dict['test_loss'][-1])
    print('Final top eigenvalue:', res_dict['top_eigvals'][-1][0])
    print('Final Hessian trace:', res_dict['trace_hessian'][-1])

    # save logs and hyperparameters
    results = {
        'tot_iter': args.tot_iter,
        'train_loss': res_dict['train_loss'],
        'test_loss': res_dict['test_loss'],
        'top_eigvals': res_dict['top_eigvals'],
        'hessian_trace': res_dict['trace_hessian'],
        'U_true': U_true.cpu().detach().numpy(),
        'U_final': model.U.detach().cpu().numpy()
    }

    # save hparams and results as .json and .pkl
    hparams = json.dumps(vars(args))
    with open(os.path.join(args.save_path, "hparams.json"), "w") as f:
        f.write(hparams)

    with open(os.path.join(args.save_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MatSen: Low-Rank Matrix Sensing with Gradient Descent"
    )
    parser.add_argument(
        "--save-path",
        default="./checkpoint",
        type=str,
        metavar="PATH",
        help="path to save result (default: none)",
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-2, 
        help="learning rate for optimizer (default: 1e-2)"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1
    )
    parser.add_argument(
        '--log-iter', 
        type=int, 
        default=10**3
    )
    parser.add_argument(
        '--tot-iter', 
        type=int, 
        default=10**8
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=25,
        help="batch size for training (default: 25)"
    )
    parser.add_argument(
        '--n-dim',
        type=int,
        default=20,
        help="dimension of the low-rank matrix nxn (default: 20)"
    )
    parser.add_argument(
        '--rank',
        type=int,
        default=5,
        help="rank of the low-rank matrix (default: 5)"
    )
    parser.add_argument(
        '--m-samples', 
        type=int, 
        default=100, 
        help="total number of samples (default: 100)"
    )
    parser.add_argument(
        '--m-train', 
        type=int, 
        default=25, 
        help="number of training samples (default: 1000)"
    )
    parser.add_argument(
        '--noise-rate',
        type=float,
        default=0.0,
        help="rate of noise to be added to the train labels (default: 0.0, no noise)"
    )
    parser.add_argument(
        '--sigma', 
        type=float, 
        default=0, 
        help="add noise to labels with the amount of sigma (default: 0, no noise)"
    )
    parser.add_argument(
        '--rho', 
        type=float, 
        default=0, 
        help="rho for persistent noise (default: 0, no persistent noise)"
    )
    
    args = parser.parse_args()

    args.inject_noise = False  # Default to no noise injection
    if args.sigma > 0:
        args.inject_noise = True
        
    # args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    main(args)
    