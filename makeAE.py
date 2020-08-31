import torch
import torch.nn.functional as F
import numpy as np
import pickle as pkl

import argparse
import sys

import torchvision.datasets as dset
import torchvision.transforms as transforms

def fgsm_attack(image, epsilon, data_grad):

    sign_data_grad = data_grad.sign()

    perturbed_image = image + epsilon * sign_data_grad

    return perturbed_image

def makeAE(model, test_loader, epsilon, device): 
    adv_dataset = []
    # feed the model with example 
    for data, target in test_loader:
        # send data to device
        data, target = data.to(device), target.to(device)
        
        # require gradients for input
        data.requires_grad=True

        # make output
        output = model(data)

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
