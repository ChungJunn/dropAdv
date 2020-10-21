import torch
import torch.nn.functional as F
import numpy as np
import pickle as pkl

import argparse
import sys

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

def where(cond, x, y):
    """
    code from :
    https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

'''
code adapted from https://github.com/1Konny/FGSM/blob/master/adversary.py
'''
def i_fgsm(net, x, y, criterion, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=0, x_val_max=1):
    x_adv = Variable(x.data, requires_grad=True)
    for i in range(iteration):
        h_adv = net(x_adv)
        if targeted:
            cost = criterion(h_adv, y)
        else:
            cost = -criterion(h_adv, y)

        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - alpha*x_adv.grad
        x_adv = where(x_adv > x+eps, x+eps, x_adv)
        x_adv = where(x_adv < x-eps, x-eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    h = net(x)
    h_adv = net(x_adv)

    return x_adv, h_adv, h

'''
code adopted from pytorch.org
'''
def fgsm_attack(image, epsilon, data_grad):

    sign_data_grad = data_grad.sign()

    perturbed_image = image + epsilon * sign_data_grad

    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

def makeAE_i_fgsm(model, test_loader, epsilon, alpha, iteration, device):
    model.eval()

    adv_dataset = []
    # feed the model with example 
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # check if model correctly guesses 
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        # obtain i-fgsm adversary
        ae, h_adv, h = i_fgsm(model, data, target, criterion=F.nll_loss, targeted=False, eps=epsilon, alpha=alpha, iteration=iteration)
        ae = ae.detach().cpu().numpy()

        adv_dataset.append([ae, target.item()])

    return adv_dataset 

def makeAE(model, test_loader, epsilon, device):
    model.eval() # dropout not working as adv ex is generated
    #model.train() # dropout in working as adv ex is generated

    adv_dataset = []
    # feed the model with example 
    for data, target in test_loader:
        # send data to device
        data, target = data.to(device), target.to(device)
        
        # require gradients for input
        data.requires_grad=True

        # make output
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        # compute loss
        loss = F.nll_loss(output, target.type(torch.int64))

        # backprop gradient
        model.zero_grad()
        loss.backward()

        # use gradient to obtain advEx
        data_grad = data.grad.data
        ae = fgsm_attack(data, epsilon, data_grad).detach().cpu().numpy()

        adv_dataset.append([ae, target.item()])
    
    return adv_dataset

if __name__ == '__main__':
    '''
    args: dataloder, pretrained model, epsilon
    out: adversarial examples in csv format
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, help='', default=0)
    parser.add_argument('--model_file', type=str, help='', default=0)
    parser.add_argument('--out_file', type=str, help='', default=0)
    args = parser.parse_args()
    from cifar10 import CIFAR10_CNN_model

    # load pretrained model
    model = torch.load(args.model_file)
    batch_size = 1
    device = torch.device('cuda')

    # load dataloader
    cifar_test = dset.CIFAR10("./data", train=False,
                              transform=transforms.ToTensor(),
                              target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=batch_size,
                                      shuffle=False,num_workers=2,drop_last=True)

    adv_dataset = makeAE(model, test_loader, args.epsilon, device)

    # save the data
    # save as pkl
    with open(args.out_file, 'wb') as fp:
        pkl.dump(adv_dataset, fp)
