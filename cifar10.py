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
from makeAE import makeAE, makeAE_i_fgsm 
from adv_utils import adv_train1, adv_train2, adv_test

import warnings
warnings.filterwarnings('ignore')
import os
import neptune

import sys
sys.path.insert(1, '/home/chl/wide-resnet.pytorch/networks')

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

def load_dataset(dataset):
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
        
def fgsm_test(model, testset, epsilon, device, out_file, neptune):
    test_loader_ = torch.utils.data.DataLoader(testset,batch_size=1,
                                      shuffle=False,num_workers=2,drop_last=True)

    # adversarial examples
    adv_test_data = makeAE(model, test_loader_, epsilon, device)

    # save into pkl file
    # filename = cnn-<trainscheme>-<eps>.pth
    import pickle as pkl
    with open(out_file, 'wb') as fp:
        pkl.dump(adv_test_data, fp)

    # dataloader
    adv_test_dataset = advDataset(adv_test_data)
    adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)

    adv_test_acc = test(model, device, adv_test_loader)
    print('white-box FGSM acc: {:.4f}'.format(adv_test_acc))
    neptune.set_property('white-box FGSM acc', adv_test_acc.item())

    return

def i_fgsm_test(model, testset, epsilon, alpha, iteration, device, neptune, dataset):

    datasets = torch.utils.data.random_split(testset, [1000, 9000], torch.Generator().manual_seed(42))
    testset = datasets[0]

    test_loader_ = torch.utils.data.DataLoader(testset,batch_size=1,
                                      shuffle=False,num_workers=2,drop_last=True)

    if dataset == 'mnist':
        adv_test_data = makeAE_i_fgsm(model, test_loader_, args.epsilon, alpha=alpha, iteration=iteration,device=device, x_val_min=0, x_val_max=1)
    elif dataset == 'cifar10':
        adv_test_data = makeAE_i_fgsm(model, test_loader_, args.epsilon, alpha=alpha, iteration=iteration,device=device, x_val_min=-1, x_val_max=1)
    adv_test_dataset = advDataset(adv_test_data)
    adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, drop_last=True)

    adv_test_acc = test(model, device, adv_test_loader)
    print('white-box I-FGSM acc: {:.4f}'.format(adv_test_acc))
    neptune.set_property('white-box I-FGSM acc', adv_test_acc.item())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='', default=0)
    parser.add_argument('--num_epochs', type=int, help='', default=0)
    parser.add_argument('--batch_size', type=int, help='', default=0)
    parser.add_argument('--epsilon', type=float, help='', default=0)
    parser.add_argument('--iteration', type=int, help='', default=0)
    parser.add_argument('--drop_p', type=float, help='', default=0)
    parser.add_argument('--alpha', type=float, help='', default=0)
    parser.add_argument('--patience', type=int, help='', default=0.0)
    parser.add_argument('--adv_train', type=int, help='', default=0)
    parser.add_argument('--name', type=str, help='', default=0)
    parser.add_argument('--tag', type=str, help='', default=0)
    parser.add_argument('--is_dnn', type=int, help='', default=0)
    parser.add_argument('--dataset', type=str, help='', default='')
    parser.add_argument('--model', type=str, help='', default='')
    parser.add_argument('--use_mydropout', type=int, help='', default='')

    parser.add_argument('--adv_test_out_path', type=str, help='', default=None)
    parser.add_argument('--adv_test_path1', type=str, help='', default=None)
    parser.add_argument('--adv_test_path2', type=str, help='', default=None)
    parser.add_argument('--adv_test_path3', type=str, help='', default=None)
    parser.add_argument('--adv_test_path4', type=str, help='', default=None)
    parser.add_argument('--adv_test_path5', type=str, help='', default=None)
    parser.add_argument('--adv_test_path6', type=str, help='', default=None)
    parser.add_argument('--adv_test_path7', type=str, help='', default=None)
    parser.add_argument('--adv_test_path8', type=str, help='', default=None)
    parser.add_argument('--adv_test_path9', type=str, help='', default=None)
    parser.add_argument('--load_adv_test', type=int, help='', default='')
    parser.add_argument('--seed', type=int, help='the code will generate it automatically', default=0)
    args = parser.parse_args()

    # random seed
    import random
    seed = random.randint(0, 50000)
    args.seed = seed

    params = vars(args)

    neptune.init('cjlee/dropAdv2')
    experiment = neptune.create_experiment(name=args.name, params=params)
    neptune.append_tag(args.tag)
    args.name = experiment.id

    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training parameters
    learning_rate=args.lr
    num_epoch=args.num_epochs
    torch.manual_seed(seed)
    out_file = args.name + '.pth'

    # load datasets
    train_loader, valid_loader, test_loader = load_dataset(args.dataset)
   
    if args.dataset == 'mnist':
        from adv_model import MNIST_LeNet_plus, MNIST_modelB, MNIST_modelA
        if args.model == 'lenet':
            model = MNIST_LeNet_plus(drop_p=args.drop_p, use_mydropout=args.use_mydropout).to(device)
        elif args.model == 'modelB':
            model = MNIST_modelB().to(device)
        elif args.model == 'modelA':
            model = MNIST_modelA().to(device)
        else:
            print('model must be lenet, modelB, or modelA')
            import sys; sys.exit(0)
    elif args.dataset == 'cifar10':
        # select model
        if args.model == 'base':
            model = CIFAR10_CNN_model(drop_p=args.drop_p).to(device)
        elif args.model == 'small':
            model = CIFAR10_CNN_small(drop_p=args.drop_p).to(device)
        elif args.model == 'large':
            model = CIFAR10_CNN_large(drop_p=args.drop_p).to(device)
        elif args.model == 'wide-resnet':
            from wide_resnet import Wide_ResNet
            model = Wide_ResNet(28, 10, args.drop_p, 10).to(device)
        elif args.model == 'vgg':
            from vggnet import VGG
            model = VGG(depth=11, num_classes=10).to(device)
        elif args.model == 'resnet':
            from resnet import ResNet 
            model = ResNet(depth=18, num_classes=10).to(device)
        elif args.is_dnn == 1:
            model = CIFAR10_DNN_model(drop_p=args.drop_p).to(device)
        else:
            print('model must be base, small, large, wide-resnet, vgg, or resnet')
            import sys; sys.exit(0)
    else:
        print('model must be mnist or cifar10')
        import sys; sys.exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    bc = 0
    patience = args.patience
    best_val_loss = None

    # start training
    print('=' * 90)
    print(model)
    print('=' * 90)
    print(optimizer)
    print('=' * 90)

    epoch = 1

    # train normal model
    for epoch in range(1, num_epoch + 1):
        if args.adv_train == 0:
            train_loss = train(model, device, train_loader, optimizer, epoch)
        elif args.adv_train == 1:
            train_loss = adv_train1(model, device, train_loader, optimizer, epoch, epsilon=args.epsilon, alpha=args.alpha)
        elif args.adv_train == 2:
            train_loss = adv_train2(model, device, train_loader, optimizer, epoch, epsilon=args.epsilon, alpha=args.alpha)
        else:
            print('adv_train must be 0, 1, or 2')
            import sys; sys.exit(-1)

        val_loss = validate(model, device, valid_loader)
        print('epoch {:d} | tr_loss: {:.4f} | val_loss {:.4f}'.format(epoch, train_loss, val_loss))
        neptune.log_metric('train_loss', epoch, train_loss)
        neptune.log_metric('valid_loss', epoch, val_loss)

        # see if val_acc improves
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            bc = 0
            torch.save(model, './result/' + out_file)

        # if not improved
        else:
            bc += 1
            if bc >= args.patience:
                break

    # test the model on clan examples  
    model = torch.load('./result/' + out_file)

    test_acc = test(model, device, test_loader)
    print('clean acc: {:.4f}'.format(test_acc))
    neptune.set_property('clean acc', test_acc.item())

    #### normal training ends ####
    # generate or load adversarial examples
    if args.dataset == 'mnist':
        testset = dset.MNIST("./data", train=False,
                              transform=transforms.ToTensor(),
                              target_transform=None, download=True)
    elif args.dataset == 'cifar10':
        testset = dset.CIFAR10("./data", train=False,
                              transform=transforms.ToTensor(),
                              target_transform=None, download=True)
    else:
        print('dataset must be mnist or cifar10')
        import sys; sys.exit(0)

    fgsm_test(model, testset, args.epsilon, device, args.adv_test_out_path, neptune)
    i_fgsm_test(model, testset, args.epsilon, alpha=args.alpha, iteration=args.iteration, device=device, neptune=neptune, dataset=args.dataset)

    # load dataset
    import pickle as pkl
    if args.load_adv_test == 1:
        for i in range(1,10):
            if eval('args.adv_test_path' + str(i)) is not None:
                with open(eval('args.adv_test_path' + str(i)), 'rb') as fp:
                    adv_test_data = pkl.load(fp)

                # dataloader
                adv_test_dataset = advDataset(adv_test_data)
                adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

                # run test on adversarial examples
                adv_test_acc = test(model, device, adv_test_loader)
                
                # obtain the last part of the string
                mystring = eval('args.adv_test_path' + str(i))
                mystring = mystring.split('/')[-1]

                # use it as input for netpune
                print(mystring + ' test acc: {:.4f}'.format(adv_test_acc))
                neptune.set_property(mystring, adv_test_acc.item())
