import torch
import torch.nn.functional as F
import numpy as np
import pickle as pkl

import argparse
import sys

import six.moves.cPickle as pickle 
import gzip
import warnings
warnings.filterwarnings('ignore')
import os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import torch.nn as nn

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

def makeAE_i_fgsm(model, test_loader, epsilon, alpha, iteration, device, x_val_min, x_val_max):
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
        ae, h_adv, h = i_fgsm(model, data, target, criterion=F.nll_loss, targeted=False, eps=epsilon, alpha=alpha, iteration=iteration, x_val_min=x_val_min, x_val_max=x_val_max)
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
'''
if __name__ == '__main__':
    #args: dataloder, pretrained model, epsilon
    #out: adversarial examples in csv format
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
'''

class mnist_dataloader:
    def __init__(self, batch_size, seed, tvt):
        tr_data, val_data, test_data = load_data('../mnist.pkl.gz')

        self.tr_x, self.tr_y = tr_data
        self.val_x, self.val_y = val_data
        self.test_x, self.test_y = test_data
        
        self.tr_x = self.tr_x.reshape(-1, 28, 28)
        self.val_x = self.val_x.reshape(-1, 28, 28)
        self.test_x = self.test_x.reshape(-1, 28, 28)

        if tvt == 'train':
            self.x, self.y = self.tr_x, self.tr_y
        elif tvt == 'valid':
            self.x, self.y = self.val_x, self.val_y
        else:
            self.x, self.y = self.test_x, self.test_y

        np.random.seed(seed)
        np.random.shuffle(self.x)
        np.random.shuffle(self.y)

        self.idx = 0
        self.data_len = len(self.x)
        self.batch_size = batch_size
        
    def get_minibatch(self):
        end_of_data = None
            
        rr = list(range(self.idx, self.idx+self.batch_size))

        xs = self.x[rr].reshape(self.batch_size,1,28,28) # bsz x 784
        ys = self.y[rr] # bsz x 784

        self.idx += self.batch_size
        if self.idx + self.batch_size >= self.data_len:
            end_of_data = True

        '''
        import imageio 
        samp_img = xs[0].reshape((28,28))
        imageio.imwrite('mnist_sample.jpg', samp_img)
        '''

        return xs, ys, end_of_data 

    def reset(self):
        self.idx = 0

class MNIST_CNN_model(nn.Module):
    def __init__(self):
        super(MNIST_CNN_model, self).__init__()
        self.conv = nn.Sequential(
            # conv layer 1
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            # conv layer 2
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        conv_size = self.get_conv_size((1, 28, 28)) # tensor of a MNIST image
        
        self.fc = nn.Sequential(
            nn.Linear(conv_size, 500), # conv_size = 4*4*50
            nn.Linear(500, 10)
        )
    
    def get_conv_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size, c, h, w = x.data.size() # 32*1*28*28
        x = self.conv(x)
        x = x.view(batch_size, -1) # conv_size = 4*4*50
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    
    copied from http://deeplearning.net/ and revised by hchoi
    '''

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    
    return train_set, valid_set, test_set

if __name__ == '__main__':
    device = torch.device('cuda')
    net = torch.load('../tutorials/practice_code/mnist_cnn.pth').to(device)

    bsz = 1
    seed = 25

    from torchvision import datasets, transforms
    import torch
    import torch.utils

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./tutorials/practice_code/data', train=False,
        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bsz, shuffle=True)
    

    # test model's accuracy
    '''
    preds = []
    targets = []
   
    for xs, ys in test_loader:
        xs, ys = torch.tensor(xs).type(torch.float32), torch.tensor(ys).type(torch.int64)
        xs, ys = xs.to(device), ys.to(device)
        
        h = net(xs)

        preds.append(torch.argmax(h, axis=-1)) 
        targets.append(ys.view(-1,1))

    preds = torch.cat(preds, axis=0).detach().cpu().numpy()
    targets = torch.cat(targets, axis=0).detach().cpu().numpy()

    from sklearn.metrics import accuracy_score
    print('accuracy: ', accuracy_score(targets, preds))
    '''
    import neptune
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    neptune.init('cjlee/dropAdv')
    experiment = neptune.create_experiment(name='log_mnist_image')
    neptune.append_tag('tag')
    
    for x, y in test_loader:
        x, y = torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.int64)
        x, y = x.to(device), y.to(device)
        criterion = F.nll_loss

        x_adv, h_adv, h = i_fgsm(net, x, y, criterion, targeted=False, eps=0.3, alpha=0.01, iteration=40, x_val_min=0, x_val_max=1)

        print('label')
        print(y)

        samp_img = x[0].detach().cpu().numpy().reshape((28,28))
        fig = plt.figure()
        plt.imshow(samp_img, cmap='gray')
        neptune.log_image('original image', fig)
        plt.clf()

        print('x logits')
        print(torch.exp(h))
        print(torch.argmax(h, dim=-1))

        print('x_adv logits')
        print(torch.exp(h_adv))
        print(torch.argmax(h_adv, dim=-1))

        samp_img = x_adv[0].detach().cpu().numpy().reshape((28,28))
        fig = plt.figure()
        plt.imshow(samp_img, cmap='gray')
        neptune.log_image('perturbed image', fig)
        plt.clf()

        break


    '''

    x_adv, h_adv, h = i_fgsm(net, x, y, criterion, targeted=False, eps=0.3, alpha=0.3, iteration=1, x_val_min=0, x_val_max=1)

    print('label')
    print(y)

    print('x logits')
    print(h)
    print(torch.argmax(h, dim=-1))

    print('x_adv logits')
    print(h_adv)
    print(torch.argmax(h_adv, dim=-1))
    '''
