# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')
import os
import sys

from adv_model import MNIST_LeNet_plus

def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch_idx,(data,target) in enumerate(train_loader):
        # implement training loop
        # send tensors to GPU
        data, target = data.to(device), target.to(device)
        #data = torch.flatten(data, start_dim=1)
    
        # initialize optimizer
        optimizer.zero_grad()

        # put data into model
        output = model(data)

        # compute loss
        loss = F.nll_loss(output, target)
        total_loss += loss.item()
    
        # backpropagate error using loss tensor
        loss.backward()

        # update model parameter using optimizer
        optimizer.step()

    return total_loss / batch_idx

# initialize model and training parameters
if __name__ == '__main__':
    import argparse
    from fgsm_tutorial import test

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='')
    parser.add_argument('--momentum', type=float, help='')
    parser.add_argument('--max_epochs', type=int, help='')
    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--savepath', type=str, help='')
    parser.add_argument('--epsilon', type=float, help='')
    parser.add_argument('--use_scheduler', type=int, help='')
    parser.add_argument('--step_size', type=int, help='')
    parser.add_argument('--gamma', type=float, help='')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    model = MNIST_LeNet_plus(0.0,0).to(device) 

    # learning rate scheduling
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.use_scheduler == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])), 
            batch_size=args.batch_size, shuffle=True)

    train_loss = 0.0
    # define train function
    for ei in range(args.max_epochs):
        train_loss = train(model, device, train_loader, optimizer) 
        if args.use_scheduler == 1:
            scheduler.step()
            print(optimizer)

        print('{} | {:.3f}'.format(ei+1, train_loss))
        
        torch.save(model.state_dict(), args.savepath)

    # test
    test_loader = torch.utils.data.DataLoader(
        dset.MNIST('./data', train=false, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])), 
            batch_size=1, shuffle=True)

    acc, ex = test(model, device, test_loader, epsilon=0.0)
    print('eps: 0.0 acc: ', acc)

    acc, ex = test(model, device, test_loader, epsilon=args.epsilon)
    print('eps: ', args.epsilon, 'acc: ', acc)
    
    # connect neptune
