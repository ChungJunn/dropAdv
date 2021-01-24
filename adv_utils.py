import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

from makeAE import fgsm_attack

def adv_train1(model, device, train_loader, optimizer, epoch, epsilon, alpha):
    model.train()
    total_loss = 0.0

    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #data = torch.flatten(data, start_dim=1)
        # requires grads
        data.requires_grad = True
    
        optimizer.zero_grad()

        output = model(data)
        clean_loss = F.nll_loss(output, target)

        # call FGSM attack
        clean_loss.backward(retain_graph=True)
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # clean grad
        optimizer.zero_grad()

        # forward data to obtain adv loss
        output = model(perturbed_data)

        adv_loss = F.nll_loss(output, target)

        # combined loss backward and optimizer.step
        loss = alpha * clean_loss + (1.0 - alpha) * adv_loss

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / batch_idx

def adv_train2(model, device, train_loader, optimizer, epoch, epsilon, alpha):
    model.train()
    total_loss = 0.0
    
    ae_tm1 = None
    trg_tm1 = None
    
    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # requires grads
        data.requires_grad = True
    
        optimizer.zero_grad()

        output = model(data)
        clean_loss = F.nll_loss(output, target)

        # call FGSM attack
        clean_loss.backward(retain_graph=True)
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        optimizer.zero_grad()

        # forward data to obtain adv loss
        if ae_tm1 is not None:
            output = model(ae_tm1)
            adv_loss = F.nll_loss(output, trg_tm1)
            # combined loss backward and optimizer.step
            loss = alpha * clean_loss + (1.0 - alpha) * adv_loss
            
        else:
            loss = clean_loss

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        # reserve the adversarial example
        ae_tm1 = perturbed_data
        trg_tm1 = target

    return total_loss / batch_idx

def adv_test(model, device, test_loader, epsilon):
    correct = 0
    total = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        #data  = torch.flatten(data, start_dim=1)

        # create perturbed data
        data.requires_grad = True
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # evaluate
        output = model(perturbed_data)
        _,output_index = torch.max(output,1)
        total += target.size(0)
        correct += (output_index == target).sum().float()

    acc = correct / len(test_loader.dataset)

    return acc

def adv_validate(model, device, valid_loader, epsilon):
    loss_total = 0.0
    
    for batch_idx, (data, target) in enumerate(valid_loader):
        data, target = data.to(device), target.to(device)

        # create perturbed data
        data.requires_grad = True
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # evaluate
        output = model(perturbed_data)
        adv_loss = F.nll_loss(output, target)
        loss_total += adv_loss.item()

    avg_loss = loss_total / batch_idx

    return avg_loss

class DropoutNew(nn.Module):
    def __init__(self, p_retain_hat=0.25):
        super(DropoutNew, self).__init__()
        if p_retain_hat < 0 or p_retain_hat > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p_retain_hat = p_retain_hat

    def forward(self, input):
        if not self.training: return input
        else:
            h_max, _ = torch.max(input, dim=1, keepdim=True)

            p_retain1 = (h_max - input) / (2 * h_max)
            p_retain = p_retain1 + self.p_retain_hat

            m_drop = torch.bernoulli(p_retain)

            out = (input * m_drop) * (1.0 / torch.mean(p_retain, dim=1, keepdim=True))
            
        return out

    def __repr__(self):
        training_str = ', training:' + str(self.training)
        return self.__class__.__name__ + '(' + 'p=' + str(self.p) + training_str + ')'
