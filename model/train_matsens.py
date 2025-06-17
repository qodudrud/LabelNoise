# OS and system
import os
import gc
import argparse
from argparse import Namespace
import pickle

# utils
from functools import partial
from copy import deepcopy
from tqdm import tqdm
import time

# mathematics
import numpy as np
import random
from sklearn.metrics import r2_score

# data handling
import pandas as pd

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from pyhessian import hessian

# Torchvision
import torchvision


class MatrixSensingModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        U_init = torch.eye(args.n_dim, args.n_dim, device=args.device)
        U_init /= torch.norm(U_init, p='fro')  # Normalize to unit Frobenius norm
        self.U = nn.Parameter(U_init)  # Initialize U as a learnable parameter

    def forward(self, A):
        X = self.U @ self.U.T
        return torch.einsum('mij,ij->m', A, X)


def update_label_noise(label_noise, args):
    return args.rho * label_noise + torch.randn_like(label_noise) * args.sigma_rho


def train(model, optimizer, criterion, A_train, y_train, A_test, y_test, args, log_iter_schedule=True):
    """Train the model with noisy labels and compute Hessian eigenvalues and trace."""
    train_loss, test_loss, top_eigvals, trace_hessian = validate(model, criterion, A_train, y_train, A_test, y_test)
    model.train()
    print(f"Initial Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Top Eigenvalue: {top_eigvals[0]}, Trace: {trace_hessian}")

    train_losses, test_losses = [train_loss], [test_loss]
    top_eigvals_res, trace_hessian_res = [top_eigvals], [trace_hessian]

    label_noise = torch.randn(args.m_train, device=args.device) * args.sigma_rho
    for t in range(1, args.tot_iter + 1):
        # Update label noise
        label_noise = update_label_noise(label_noise, args)
        y_train_noisy = y_train + label_noise

        # Sample a batch of data
        batch_idx = torch.randint(0, args.m_train, (args.batch_size,), device=args.device)

        A_batch = A_train[batch_idx]
        y_batch = y_train_noisy[batch_idx]

        optimizer.zero_grad()
        loss = criterion(model(A_batch), y_batch)
        loss.backward()
        optimizer.step()

        # Logging
        if t % args.log_iter == 0:  # Optional: log less frequently
            train_loss, test_loss, top_eigvals, trace_hessian = validate(model, criterion, A_train, y_train, A_test, y_test)
            model.train()

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            top_eigvals_res.append(top_eigvals)
            trace_hessian_res.append(trace_hessian)
            print(f"Iter {t}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Top Eigenvalue: {top_eigvals[0]}, Trace: {trace_hessian}")
            
            # Optional: update the log_iter dynamically
            if log_iter_schedule == True and t > 0 and t % (args.log_iter * 10) == 0:
                args.log_iter *= 10  # Increase log_iter to log less frequently

    return {'train_loss': train_losses, 'test_loss': test_losses, 'top_eigvals': top_eigvals_res, 'trace_hessian': trace_hessian_res}


def validate(model, criterion, A_train, y_train, A_test=None, y_test=None):
    """Validate the model and compute Hessian eigenvalues and trace."""
    if A_test is not None and y_test is not None:
        assert A_test.shape[0] == y_test.shape[0], "Test data and labels must have the same number of samples."

    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(A_train), y_train).item()

    # Compute Hessian eigenvalues and trace
    hessian_comp = hessian(model, criterion, data=(A_train, y_train), cuda=True)
    top_eigvals, _ = hessian_comp.eigenvalues(top_n=5)
    trace_hessian = np.mean(hessian_comp.trace())

    if A_test is not None and y_test is not None:
        with torch.no_grad():
            test_loss = criterion(model(A_test), y_test).item()
        return train_loss, test_loss, np.array(top_eigvals), trace_hessian
    else:
        return train_loss, None, np.array(top_eigvals), trace_hessian
