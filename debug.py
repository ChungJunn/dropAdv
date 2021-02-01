import torch
import torch.nn as nn

import torchvision.datasets as dset
import torchvision.transforms as transforms

from adv_data import advDataset
from adv_model import MNIST_LeNet_plus
from makeAE import makeAE
from cifar10 import test

device = torch.device("cuda")

# make advDataset
testset = mnist_test = dset.MNIST("./data", train=False,
                                    transform=transforms.ToTensor(),
                                    target_transform=None, download=True)

testloader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=False, num_workers=2, drop_last=True)

# load a model
# import lenet model 
model = MNIST_LeNet_plus(drop_p=0.0, use_mydropout=0).to(device)

# makeAE
adv_test_data = makeAE(model, testloader, epsilon=0.3, device=device)

# make advdataloader
adv_test_dataset = advDataset(adv_test_data)

adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=100, shuffle=True, num_workers=2, drop_last=True)

# do FGSM test and pdb.set_trace()
# reproduce the error 
acc = test(model, device, adv_test_loader)
print("acc: ", acc)
