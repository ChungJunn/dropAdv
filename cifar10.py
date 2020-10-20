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
from makeAE import makeAE

import warnings
warnings.filterwarnings('ignore')
import os
import neptune

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

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, help='', default=0)
    parser.add_argument('--num_epochs', type=int, help='', default=0)
    parser.add_argument('--batch_size', type=int, help='', default=0)
    parser.add_argument('--epsilon', type=float, help='', default=0)
    parser.add_argument('--drop_p', type=float, help='', default=0)
    parser.add_argument('--alpha', type=float, help='', default=0)
    parser.add_argument('--patience', type=int, help='', default=0.0)
    parser.add_argument('--adv_train', type=int, help='', default=0)
    parser.add_argument('--name', type=str, help='', default=0)
    parser.add_argument('--tag', type=str, help='', default=0)
    parser.add_argument('--is_dnn', type=int, help='', default=0)
    parser.add_argument('--dataset', type=str, help='', default='')
    parser.add_argument('--model', type=str, help='', default='')

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
    parser.add_argument('--adv_test_path10', type=str, help='', default=None)
    parser.add_argument('--load_adv_test', type=int, help='', default='')
    parser.add_argument('--seed', type=int, help='the code will generate it automatically', default=0)
    args = parser.parse_args()

    # random seed
    import random
    seed = random.randint(0, 50000)
    args.seed = seed

    params = vars(args)

    neptune.init('cjlee/dropAdv')
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

    if args.dataset == 'mnist':
        # MNIST dataset
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

        from adv_model import MNIST_LeNet_plus, MNIST_modelB, MNIST_modelA
        if args.model == 'lenet':
            model = MNIST_LeNet_plus(drop_p=args.drop_p).to(device)
        elif args.model == 'modelB':
            model = MNIST_modelB().to(device)
        elif args.model == 'modelA':
            model = MNIST_modelA().to(device)
        else:
            print('model must be lenet, modelB, or modelA')
            import sys; sys.exit(0)

    elif args.dataset == 'cifar10':
        # CIFAR10 dataset
        # reload valid and testloader with batch_size
        cifar_train = dset.CIFAR10("./data", train=True,
                                   transform=transforms.ToTensor(),
                                   target_transform=None, download=True)
        cifar_test = dset.CIFAR10("./data", train=False,
                                  transform=transforms.ToTensor(),
                                  target_transform=None, download=True)
        from adv_model import CIFAR10_CNN_small
        from adv_model import CIFAR10_CNN_large

        # create valid dataset
        datasets = torch.utils.data.random_split(cifar_train, [45000, 5000], torch.Generator().manual_seed(42)) # do not change manual_seed (the advvalidset has been created with this manual seed)

        cifar_train, cifar_valid = datasets[0], datasets[1]
        
        train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=batch_size,
                                          shuffle=True,num_workers=2,drop_last=True)
        valid_loader = torch.utils.data.DataLoader(cifar_valid,batch_size=batch_size,
                                          shuffle=True,num_workers=2,drop_last=True)
        test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=batch_size,
                                          shuffle=False,num_workers=2,drop_last=True)
        # select model
        if args.model == 'base':
            model = CIFAR10_CNN_model(drop_p=args.drop_p).to(device)
        elif args.model == 'small':
            model = CIFAR10_CNN_small(drop_p=args.drop_p).to(device)
        elif args.model == 'large':
            model = CIFAR10_CNN_large(drop_p=args.drop_p).to(device)
        elif args.is_dnn == 1:
            model = CIFAR10_DNN_model(drop_p=args.drop_p).to(device)

    else:
        print("data must be either mnist or cifar10")
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

    # generate adversarial training and test sets  
    model = torch.load('./result/' + out_file)

    test_acc = test(model, device, test_loader)
    print('clean acc: {:.4f}'.format(test_acc))
    neptune.set_property('clean acc', test_acc.item())

    #### normal training ends ####
    # generate or load adversarial examples
    if args.load_adv_test == 0:
        test_loader_ = torch.utils.data.DataLoader(mnist_test,batch_size=1, #TODO change name of testset
                                          shuffle=False,num_workers=2,drop_last=True)

        # adversarial examples
        adv_test_data = makeAE(model, test_loader_, args.epsilon, device)

        # save into pkl file
        # filename = cnn-<trainscheme>-<eps>.pth
        import pickle as pkl
        with open(args.adv_test_out_path, 'wb') as fp:
            pkl.dump(adv_test_data, fp)

        # dataloader
        adv_test_dataset = advDataset(adv_test_data)
        adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2, drop_last=True)

        adv_test_acc = test(model, device, adv_test_loader)
        print('white-box FGSM acc: {:.4f}'.format(adv_test_acc))
        neptune.set_property('white-box FGSM acc', adv_test_acc.item())

    # load dataset
    import pickle as pkl
    if args.load_adv_test == 1:
        for i in range(1,11):
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

