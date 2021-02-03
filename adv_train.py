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
import neptune

from adv_model import MNIST_LeNet_plus
from fgsm_tutorial import ifgsm_test

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
    from fgsm_tutorial import fgsm_test

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='')
    parser.add_argument('--momentum', type=float, help='')
    parser.add_argument('--max_epochs', type=int, help='')
    parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--savepath', type=str, help='')
    parser.add_argument('--epsilon', type=float, help='')
    parser.add_argument('--iteration', type=int, help='')
    parser.add_argument('--use_scheduler', type=int, help='')
    parser.add_argument('--step_size', type=int, help='')
    parser.add_argument('--gamma', type=float, help='')
    parser.add_argument('--name', type=str, help='')
    parser.add_argument('--tag', type=str, help='')
    parser.add_argument('--weight_decay', type=float, help='')
    args = parser.parse_args()
    params = vars(args)

    # add args for ifgsm
    # import related items
    # test the code
    
    device = torch.device('cuda')
    model = MNIST_LeNet_plus(0.0,0).to(device) 

    neptune.init('cjlee/dropAdv2')
    experiment = neptune.create_experiment(name=args.name, params=params)
    neptune.append_tag(args.tag)
    args.name = experiment.id
    args.savepath = './result/' + args.name + '.pth'

    # learning rate scheduling
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.use_scheduler == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])), 
            batch_size=args.batch_size, shuffle=True)
    x_val_max = 1.0
    x_val_min = 0.0

    train_loss = 0.0
    # define train function
    for ei in range(args.max_epochs):
        train_loss = train(model, device, train_loader, optimizer) 
        if args.use_scheduler == 1:
            scheduler.step()

        print('{} | {:.3f}'.format(ei+1, train_loss))
        neptune.log_metric('train_loss', ei, train_loss)
        
        torch.save(model.state_dict(), args.savepath)

    # test
    test_loader = torch.utils.data.DataLoader(
        dset.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                ])), 
            batch_size=1, shuffle=True)

    acc, ex = fgsm_test(model, device, test_loader, epsilon=0.0)
    print('eps: 0.0 acc: ', acc)
    neptune.set_property('clean acc', acc)

    acc, ex = fgsm_test(model, device, test_loader, epsilon=args.epsilon)
    print('eps: ', args.epsilon, 'acc: ', acc)
    neptune.set_property('fgsm acc', acc)

    ifgsm_alpha = args.epsilon / float(args.iteration)
    print('ifgsm_alpha: {:.4f}'.format(ifgsm_alpha))
    acc = ifgsm_test(model, test_loader, args.epsilon, ifgsm_alpha, args.iteration, device, x_val_min, x_val_max)
    neptune.set_property('ifgsm acc', acc)
