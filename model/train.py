import gc

import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import time


class NoiseInjector:
    def __init__(self, model, args):
        self.device = args.device
        self.eps = args.perturb_eps
        self.perturb_type = args.perturb_type
        self.lr = args.lr
        
        self.std = self.eps * np.sqrt(self.lr) # Standard deviation for Gaussian noise
        
        self.param_shapes = [p.shape for p in model.parameters() if p.requires_grad]
        self.total_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if args.perturb_type == 'anti':
            self.eps0 = args.perturb_eps0
            self.std = self.eps0 * np.sqrt(self.lr) # Standard deviation for Gaussian noise

            self.tau = args.perturb_tau
            self.anti_std = (self.eps/self.tau) * np.sqrt(self.lr)

            stationary_var = (self.eps ** 2 / self.tau) / (2 - self.lr / self.tau)
            self.anti_noise = torch.randn(self.total_size, device=self.device) * np.sqrt(stationary_var)
            self.anti_noise = self.anti_noise.to(self.device)

    def _update_anti_noise(self):
        with torch.no_grad():
            return (1 - (self.lr/self.tau)) * self.anti_noise + torch.randn(self.anti_noise.size()).to(self.device) * self.anti_std

    def generate_noise(self):
        if self.perturb_type == 'random':
            return torch.randn(self.total_size, device=self.device) * self.std
        elif self.perturb_type == 'anti':
            next_anti_noise = self._update_anti_noise()
            inj_noise = self.tau * (next_anti_noise - self.anti_noise) # add colored noise
            inj_noise += torch.randn(self.total_size, device=self.device) * self.std # add Gaussian noise
            self.anti_noise = next_anti_noise
            return inj_noise

    def apply_noise(self, model):
        inj_noise = self.generate_noise()

        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    num = p.numel()
                    p.add_(inj_noise[offset:offset+num].view_as(p))
                    offset += num


def train_gd(model, train_loader, optim, args, noise_injector = None, save_acc = False, epoch = -1):
    """
        Train the model for one epoch.
        Args:
            model: the model to be trained
            train_loader: the dataloader for training data
            optim: the optimizer for the model
            args: the arguments for training
            perturb_model: a function to perturb the model (optional)
        Returns:
            train_log: a dictionary containing the training loss and accuracy (if save_acc is True)
    """
    assert args.optim == 'gd', 'train_gd() only supports GD optimizer.'

    train_log = {}
    if epoch >= 0:
        train_log['epoch'] = []
    train_log['loss'] = []
    train_log['acc'] = []

    model.train()
    optim.zero_grad()

    # # perturb the model
    # if noise_injector is not None:
    #     if args.perturb_eps != 0:
    #         noise_injector.apply_noise(model)

    for i, data in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):
            inputs, labels = data[0].to(args.device), data[1].to(args.device).long()
            outputs = model(inputs)

            if args.loss_type == 'ce':
                criterion = nn.CrossEntropyLoss(reduction='sum')
                loss = criterion(outputs, labels)
                loss = loss/len(train_loader.dataset) # normalize loss by dataset size
                loss.backward()

    optim.step()

    # perturb the model
    if noise_injector is not None:
        if args.perturb_eps != 0:
            noise_injector.apply_noise(model)

    # save the training accuracy
    if save_acc:
        # calculate acc.
        _, pred = torch.max(outputs.data, 1)
        correct = (pred == labels).sum()
        total = labels.size(0)
        acc = (100*correct/total).detach().cpu().numpy()
    else:
        acc = 0

    if epoch >= 0:
        train_log['epoch'].append(epoch)
    train_log['loss'].append(loss.detach().cpu().numpy())
    train_log['acc'].append(acc)
    
    return train_log


def train(model, train_loader, optim, args, noise_injector = None, save_acc = False, epoch = -1):
    """
        Train the model for one epoch.
        Args:
            model: the model to be trained
            train_loader: the dataloader for training data
            optim: the optimizer for the model
            args: the arguments for training
            perturb_model: a function to perturb the model (optional)
        Returns:
            train_log: a dictionary containing the training loss and accuracy (if save_acc is True)
    """
    train_log = {}
    if epoch >= 0:
        train_log['epoch'] = []
    train_log['loss'] = []
    train_log['acc'] = []

    if args.optim == 'gd':
        return train_gd(model, train_loader, optim, args, noise_injector=noise_injector, save_acc=save_acc, epoch=epoch)

    model.train()
    for i, data in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):
            optim.zero_grad()

            inputs, labels = data[0].to(args.device), data[1].to(args.device).long()
            outputs = model(inputs)

            if args.loss_type == 'ce':
                criterion = nn.CrossEntropyLoss(reduction='mean')
                loss = criterion(outputs, labels)

            loss.backward()
            optim.step()

            # perturb the model
            if noise_injector is not None:
                if args.perturb_eps != 0:
                    noise_injector.apply_noise(model)

            # save the training accuracy
            if save_acc:
                # calculate acc.
                _, pred = torch.max(outputs.data, 1)
                correct = (pred == labels).sum()
                total = labels.size(0)
                acc = (100*correct/total).detach().cpu().numpy()
            else:
                acc = 0

            if epoch >= 0:
                train_log['epoch'].append(epoch)
            train_log['loss'].append(loss.detach().cpu().numpy())
            train_log['acc'].append(acc)
            
    return train_log


def test(model, test_loader, args, epoch = -1):
    """
        Test the model on the test set.
        Args:
            model: the model to be tested
            test_loader: the dataloader for test data
            args: the arguments for testing
        Returns:
            test_log: a dictionary containing the test loss and accuracy
    """
    test_log = {}
    if epoch >= 0:
        test_log['epoch'] = []
    test_log['loss'] = []
    test_log['acc'] = []

    correct = 0
    total = 0
    loss = 0
    
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(args.device), data[1].to(args.device).long()
            outputs = model(inputs)
            
            # calculate acc.
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
            
            # calculate loss
            loss_ = criterion(outputs, labels)
            loss_ = loss_.sum()
            loss += loss_.detach().cpu().numpy()

    if epoch >= 0:
        test_log['epoch'].append(epoch)
    test_log['loss'].append(loss/len(test_loader.dataset))
    test_log['acc'].append((100*correct/total).detach().cpu().numpy())
                
    return test_log