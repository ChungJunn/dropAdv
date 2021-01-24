import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from test import DropoutNew as MyDropout

class MNIST_LeNet_plus(nn.Module):
    def __init__(self, drop_p, use_mydropout):
        super(MNIST_LeNet_plus,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,32,5,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        conv_size = self.get_conv_size((1,28,28))

        if use_mydropout == 0:
            self.fc_layer = nn.Sequential(
                nn.Dropout(p=drop_p),
                nn.Linear(conv_size,1024),
                nn.ReLU(),
                nn.Dropout(p=drop_p),
                nn.Linear(1024,10),
            )
        elif use_mydropout == 1:
            self.fc_layer = nn.Sequential(
                MyDropout(p=drop_p),
                nn.Linear(conv_size,1024),
                nn.ReLU(),
                MyDropout(p=drop_p),
                nn.Linear(1024,10),
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

class MNIST_modelA(nn.Module):
    def __init__(self):
        super(MNIST_modelA,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,64,5,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,5,padding=1),
            nn.ReLU(),
        )

        conv_size = self.get_conv_size((1,28,28))

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(conv_size,128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,10),
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

class MNIST_modelB(nn.Module):
    def __init__(self):
        super(MNIST_modelB,self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1,64,8,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,6,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,5,padding=1),
            nn.ReLU(),
        )

        conv_size = self.get_conv_size((1,28,28))

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(conv_size,10),
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

'''
class MNIST_modelC(nn.Module):
    def __init__(self, drop_p):
        super(MNIST_modelC,self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1,128,8,padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,3,padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2,2),
        )

        conv_size = self.get_conv_size((1,28,28))

        self.fc_layer = nn.Sequential(
            nn.Linear(conv_size,128),
            nn.ReLU(),
            nn.Linear(128,10),
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
'''

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
        x = torch.flatten(x, start_dim=1)

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
        x = torch.flatten(x, start_dim=1)

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
            nn.MaxPool2d(2,2),
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

class CIFAR10_CNN_small(nn.Module):
    def __init__(self, drop_p):
        super(CIFAR10_CNN_small,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32 x 16 x 16 (batch_size width height)
            
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 128 x 8 x 8
        )

        conv_size = self.get_conv_size((3,32,32))

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(conv_size,100),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(100,10),
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

class CIFAR10_CNN_large(nn.Module):
    def __init__(self, drop_p):
        super(CIFAR10_CNN_large,self).__init__()
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
            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        conv_size = self.get_conv_size((3,32,32))

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(conv_size,400),
            nn.ReLU(),
            nn.Dropout(p=drop_p),
            nn.Linear(400,200),
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
