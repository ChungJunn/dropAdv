import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader
from adv_data import advDataset
from adv_model import CIFAR10_CNN_model
from networks.wide_resnet import Wide_ResNet
from makeAE import makeAE, makeAE_i_fgsm 
from adv_utils import adv_train1, adv_train2, adv_test

import warnings
warnings.filterwarnings('ignore')
import os
import neptune

import sys
import random
from adv_model import MNIST_LeNet_plus

def train(model, device, train_loader, optimizer, epoch):
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

        if batch_idx == 10: break

    return total_loss / batch_idx

def validate(model, device, valid_loader):
    model.eval()
    total_loss = 0.0
    for batch_idx,(data,target) in enumerate(valid_loader):
        # implement training loop
        # send tensors to GPU
        data, target = data.to(device), target.to(device)

        # put data into model
        output = model(data)

        # compute loss
        loss = F.nll_loss(output, target)
        total_loss += loss.item()

    return total_loss / batch_idx


def test(model, device, test_loader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            _,output_index = torch.max(output,1)
            total += target.size(0)
            correct += (output_index == target).sum().float()

    acc = correct / total

    return acc

def load_dataset(dataset, batch_size):
    if dataset == 'mnist':
        mnist_train = dset.MNIST("./data", train=True,
                                   transform=transforms.ToTensor(),
                                   target_transform=None, download=True)
        mnist_test = dset.MNIST("./data", train=False,
                                  transform=transforms.ToTensor(),
                                  target_transform=None, download=True)

        # create valid dataset
        datasets = torch.utils.data.random_split(mnist_train, [54000, 6000], torch.Generator().manual_seed(42)) # do not change manual_seed (the advvalidset has been created with this manual seed)
        mnist_train, mnist_valid = datasets[0], datasets[1]

        train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,
                                          shuffle=True,num_workers=2,drop_last=True)
        valid_loader = torch.utils.data.DataLoader(mnist_valid,batch_size=batch_size,
                                          shuffle=True,num_workers=2,drop_last=True)
        test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,
                                          shuffle=False,num_workers=2,drop_last=True)

        return train_loader, valid_loader, test_loader
    
    elif args.dataset == 'cifar10':
        input_size = (32,32)
        # CIFAR10 dataset
        cifar_transforms = transforms.Compose([
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        # reload valid and testloader with batch_size
        cifar_train = dset.CIFAR10("./data", train=True,
                                   transform=cifar_transforms,
                                   target_transform=None, download=True)
        cifar_test = dset.CIFAR10("./data", train=False,
                                  transform=cifar_transforms,
                                  target_transform=None, download=True)

        # create valid dataset
        datasets = torch.utils.data.random_split(cifar_train, [45000, 5000], torch.Generator().manual_seed(42)) # do not change manual_seed (the advvalidset has been created with this manual seed)

        cifar_train, cifar_valid = datasets[0], datasets[1]
        
        train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=batch_size,
                                          shuffle=True,num_workers=2,drop_last=True)
        valid_loader = torch.utils.data.DataLoader(cifar_valid,batch_size=batch_size,
                                          shuffle=True,num_workers=2,drop_last=True)
        test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=batch_size,
                                          shuffle=False,num_workers=2,drop_last=True)
        
        return train_loader, valid_loader, test_loader

    else:
        print("data must be either mnist or cifar10")
        import sys; sys.exit(0)
        return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='', default=0)
    parser.add_argument('--num_epochs', type=int, help='', default=0)
    parser.add_argument('--batch_size', type=int, help='', default=0)
    parser.add_argument('--drop_p', type=float, help='', default=0)
    parser.add_argument('--patience', type=int, help='', default=0.0)
    parser.add_argument('--name', type=str, help='', default=0)
    parser.add_argument('--tag', type=str, help='', default=0)
    parser.add_argument('--dataset', type=str, help='', default='')
    parser.add_argument('--model', type=str, help='', default='')
    parser.add_argument('--use_mydropout', type=int, help='', default='')
    parser.add_argument('--use_step_policy', type=int, help='', default='')

    parser.add_argument('--seed', type=int, help='the code will generate it automatically', default=0)
    args = parser.parse_args()
    params = vars(args)

    # random seed
    args.seed = random.randint(0, 50000)
    torch.manual_seed(args.seed)

    neptune.init('cjlee/dropAdv2')
    experiment = neptune.create_experiment(name=args.name, params=params)
    neptune.append_tag(args.tag)
    args.name = experiment.id
    out_file = args.name + '.pth'

    # load datasets
    train_loader, valid_loader, test_loader = load_dataset(args.dataset, args.batch_size)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == 'lenet':
        model = MNIST_LeNet_plus(drop_p=args.drop_p, use_mydropout=args.use_mydropout).to(device)
    elif args.model == 'wide-resnet':
        model = Wide_ResNet(28, 10, args.drop_p, 10, args.use_mydropout).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.use_step_policy == 1:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        '''
        based on https://towardsdatascience.com/learning-rate-schedules-and-adaptive-
        learning-rate-methods-for-deep-learning-2c8f433990d1
        '''

    bc = 0
    patience = args.patience
    best_val_loss = None

    # train normal model
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)

        val_loss = validate(model, device, valid_loader)
        print('epoch {:d} | tr_loss: {:.4f} | val_loss {:.4f}'.format(epoch, train_loss, val_loss))
        if args.use_step_policy == 1:
            scheduler.step() 
            print('scheduler and optimizer info: ')
            print(scheduler)
            print(optimizer)

        if neptune is not None:
            neptune.log_metric('train_loss', epoch, train_loss)
            neptune.log_metric('valid_loss', epoch, val_loss)

        # see if val_acc improves
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            bc = 0
            torch.save(model, './result/' + out_file)
        else:
            bc += 1
            if bc >= args.patience:
                break

    # test the model on clean examples  
    model = torch.load('./result/' + out_file)

    test_acc = test(model, device, test_loader)
    print('clean acc: {:.4f}'.format(test_acc))
    if neptune is not None: neptune.set_property('clean acc', test_acc.item())
