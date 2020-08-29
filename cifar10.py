import os
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
import neptune

name = 'test'
tag = 'tag'

#neptune.init('cjlee/dropAdv')
#neptune.create_experiment(name=name)
#neptune.append_tag(tag)

print(torch.__version__)

class CIFAR10_CNN_model(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN_model,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32 x 16 x 16 (batch_size width height)
            
            # define 2 additional convolution layers and maxpool layer
            # add one conv layer
            nn.Conv2d(32,64,3,padding=1),
            # add activation function
            nn.ReLU(),
            # add another conv layer 
            nn.Conv2d(64,128,3,padding=1),
            # add activation function
            nn.ReLU(),
            # add max pooling layer
            nn.MaxPool2d(2,2), # 128 x 8 x 8
            
            # another two additional layers
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        conv_size = self.get_conv_size((3,32,32))

        self.fc_layer = nn.Sequential(
            nn.Linear(conv_size,200),
            nn.ReLU(),
            nn.Linear(200,10)
        )       

    def get_conv_size(self, shape):
        o = self.layer(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def forward(self,x):
        # Define forward function of the model

        # obtain batch size
        batch_size, c, h, w = x.data.size()

        # feed data through conv layers
        out = self.layer(x)

        # reshape the output of convolution layer for fully-connected layer
        out = out.view(batch_size, -1)

        # feed data through feed-forward layer
        out = self.fc_layer(out)
        return F.log_softmax(out, dim=1)

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    total_loss = 0.0
    for batch_idx,(data,target) in enumerate(train_loader):
        # implement training loop
        # send tensors to GPU
        data, target = data.to(device), target.to(device)
    
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
    
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))

    return total_loss / batch_idx

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def adv_train1(model, device, train_loader, optimizer, epoch, log_interval, epsilon, alpha):
    model.train()
    total_loss = 0.0

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
    
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))

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

    acc = correct / len(test_loader.dataset)

    print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return acc

if __name__ == '__main__':
    # check model
    model = CIFAR10_CNN_model()
    print(model)
    print('=' * 90)

    # dataset
    batch_size = 32

    cifar_train = dset.CIFAR10("./data", train=True, 
                               transform=transforms.ToTensor(), 
                               target_transform=None, download=True)
    cifar_test = dset.CIFAR10("./data", train=False, 
                              transform=transforms.ToTensor(), 
                              target_transform=None, download=True)

    # create valid dataset
    datasets = torch.utils.data.random_split(cifar_train, [45000, 5000])
    cifar_train, cifar_valid = datasets[0], datasets[1]

    train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=batch_size, 
                                      shuffle=True,num_workers=2,drop_last=True)
    valid_loader = torch.utils.data.DataLoader(cifar_valid,batch_size=batch_size, 
                                      shuffle=True,num_workers=2,drop_last=True)
    test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=batch_size, 
                                      shuffle=False,num_workers=2,drop_last=True)

    print('train dataset: ', cifar_train.__getitem__(0)[0].size(), cifar_train.__len__())
    print('test dataset: ', cifar_test.__getitem__(0)[0].size(), cifar_test.__len__())
    print('=' * 90)

    for batch, (data, target) in enumerate(train_loader):
        print('data shape: ', data.shape)
        print('target shape: ', target.shape)
        break

    seed=1
    learning_rate=0.001
    num_epoch=5000
    log_interval=100

    epsilon = 0.1
    alpha = 0.5

    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10_CNN_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bc = 0
    patience = 5
    best_val_acc = None

    for epoch in range(1, num_epoch + 1):
        #train_loss = train(model, device, train_loader, optimizer, epoch, log_interval)
        train_loss = adv_train1(model, device, train_loader, optimizer, epoch, log_interval, epsilon=epsilon, alpha=alpha)
        val_acc = test(model, device, valid_loader)

        #neptune.log_metric('train_loss', epoch, train_loss)
        #neptune.log_metric('valid_acc', epoch, val_acc)

        # see if val_acc improves
        if best_val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            bc = 0
            torch.save(model, './cifar10_pretrained.pth')

        # if not improved
        else:
            bc += 1
            if bc >= patience:
                break

    model = torch.load('./cifar10_pretrained.pth')
    test_acc = test(model, device, test_loader)
    #neptune.set_property('test acc', test_acc)
