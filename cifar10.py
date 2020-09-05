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
from makeAE import makeAE

import warnings
warnings.filterwarnings('ignore')
import os
import neptune

class CIFAR10_DNN_model(nn.Module):
    def __init__(self, drop_p):
        super(CIFAR10_DNN_model,self).__init__()
        self.fc1 = nn.Linear(3072,2500)
        self.fc2 = nn.Linear(2500,2000)
        self.fc3 = nn.Linear(2000,1000)
        self.fc4 = nn.Linear(1000,100)
        self.fc5 = nn.Linear(100,10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

class CIFAR10_CNN_model(nn.Module):
    def __init__(self, drop_p):
        super(CIFAR10_CNN_model,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32 x 16 x 16 (batch_size width height)
            
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 128 x 8 x 8
            
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        conv_size = self.get_conv_size((3,32,32))

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(conv_size,200),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(200,10),
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

def eval(model, device, valid_loader):
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

def test(model, device, test_loader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #data  = torch.flatten(data, start_dim=1)
            output = model(data)

            _,output_index = torch.max(output,1)
            total += target.size(0)
            correct += (output_index == target).sum().float()

    acc = correct / len(test_loader.dataset)

    return acc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='', default=0)
    parser.add_argument('--lr', type=float, help='', default=0)
    parser.add_argument('--num_epochs', type=int, help='', default=0)
    parser.add_argument('--log_interval', type=int, help='', default=0)
    parser.add_argument('--batch_size', type=int, help='', default=0)
    parser.add_argument('--epsilon', type=float, help='', default=0)
    parser.add_argument('--drop_p', type=float, help='', default=0)
    parser.add_argument('--alpha', type=float, help='', default=0)
    parser.add_argument('--patience', type=float, help='', default=0.0)
    parser.add_argument('--name', type=str, help='', default=0)
    parser.add_argument('--tag', type=str, help='', default=0)
    args = parser.parse_args()

    params = vars(args)

    neptune.init('cjlee/dropAdv')
    neptune.create_experiment(name=args.name, params=params)
    neptune.append_tag(args.tag)

    # dataset
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # reload valid and testloader with batch_size
    cifar_train = dset.CIFAR10("./data", train=True,
                               transform=transforms.ToTensor(),
                               target_transform=None, download=True)
    cifar_test = dset.CIFAR10("./data", train=False,
                              transform=transforms.ToTensor(),
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

    # training parameters
    seed=args.seed
    learning_rate=args.lr
    num_epoch=args.num_epochs
    log_interval=args.log_interval

    torch.manual_seed(seed)
    out_file = args.name + '.pth'

    model = CIFAR10_CNN_model(drop_p=args.drop_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bc = 0
    patience = args.patience
    best_val_acc = None

    # start training
    print('=' * 90)
    print(model)
    print('=' * 90)
    print(optimizer)
    print('=' * 90)

    # train normal model
    for epoch in range(1, num_epoch + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, log_interval)
        val_acc = test(model, device, valid_loader)

        print('epoch {:d} | tr_loss: {:.4f} | val_acc {:.4f}'.format(epoch, train_loss, val_acc))
        neptune.log_metric('train_loss', epoch, train_loss)
        neptune.log_metric('valid_acc', epoch, val_acc)

        # see if val_acc improves
        if best_val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            bc = 0
            torch.save(model, './result/' + out_file)

        # if not improved
        else:
            bc += 1
            if bc >= patience:
                break

    # generate adversarial training and test sets  
    model = torch.load('./result/' + out_file)

    test_acc = test(model, device, test_loader)
    print('[normal train] test acc: {:.4f}'.format(test_acc))
    neptune.set_property('[normal train] test acc', test_acc.item())

    # generate adversarial examples for test and valid set 
    valid_loader_ = torch.utils.data.DataLoader(cifar_valid,batch_size=1,
                                      shuffle=True,num_workers=2,drop_last=True)
    test_loader_ = torch.utils.data.DataLoader(cifar_test,batch_size=1,
                                      shuffle=False,num_workers=2,drop_last=True)

    adv_val_data = makeAE(model, valid_loader_, args.epsilon, device)
    adv_test_data = makeAE(model, test_loader_, args.epsilon, device)
    
    # load adversarial examples
    adv_valid_dataset = advDataset(adv_val_data)
    adv_valid_loader = torch.utils.data.DataLoader(adv_valid_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)
    adv_test_dataset = advDataset(adv_test_data)
    adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)

    adv_test_acc = test(model, device, adv_test_loader)
    print('[normal train] adv_test acc: {:.4f}'.format(adv_test_acc))
    neptune.set_property('[normal train] adv_test acc', adv_test_acc.item())

    # retrain the model with adversarial training
    bc = 0
    best_val_acc = None

    model = CIFAR10_CNN_model(drop_p=args.drop_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epoch + 1):
        train_loss = adv_train1(model, device, train_loader, optimizer, epoch, log_interval, epsilon=args.epsilon, alpha=args.alpha)
        val_acc = test(model, device, adv_valid_loader)

        print('epoch {:d} | tr_loss: {:.4f} | val_acc {:.4f}'.format(epoch, train_loss, val_acc))
        neptune.log_metric('[adv_train] train_loss', epoch, train_loss)
        neptune.log_metric('[adv_train] valid_acc', epoch, val_acc)

        # see if val_acc improves
        if best_val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            bc = 0
            torch.save(model, './result/' + out_file)

        # if not improved
        else:
            bc += 1
            if bc >= patience:
                break

    # evaluate the model
    test_acc = test(model, device, test_loader)
    print('[adv_train] test acc: {:.4f}'.format(test_acc))
    neptune.set_property('[advtrain] test acc', test_acc.item())
    
    adv_test_acc = test(model, device, adv_test_loader)
    print('[adv_train] adv_test acc: {:.4f}'.format(adv_test_acc))
    neptune.set_property('[advtrain] adv_test acc', adv_test_acc.item())
