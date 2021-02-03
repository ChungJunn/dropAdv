# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

'''
code adopted from 1Konny
'''
def where(cond, x, y):
    """
    code from :
    https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

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

def ifgsm_test(model, test_loader, epsilon, alpha, iteration, device, x_val_min, x_val_max, n_generation=1000):
    if n_generation > len(test_loader):
        print("n_generation must be less than test set")
        import sys; sys.exit(-1)

    model.eval()
    correct = 0

    # feed the model with example 
    for idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        # check if model correctly guesses 
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        # obtain i-fgsm adversary
        x_adv, h_adv, h = i_fgsm(model, data, target, criterion=F.nll_loss, targeted=False, eps=epsilon, alpha=alpha, iteration=iteration, x_val_min=x_val_min, x_val_max=x_val_max)

        output = model(x_adv)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1

        if (idx+1) == n_generation: 
            print('[ifsm_test] {} adv_samples are used for test'.format(idx+1))
            break

    final_acc = correct/float(n_generation)
    print("IGFSM eps: {:.4f} iteration: {}\tTest Accuracy = {} / {} = {}".format(epsilon, iteration, correct, n_generation, final_acc))

    return final_acc 

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def fgsm_test( model, device, test_loader, epsilon ):
    model.eval()

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

if __name__ == '__main__':
    #epsilons = [0, .05, .1, .15, .2, .25, .3]
    epsilons = [0,.3]
    pretrained_model = "./mydict.pth"
    use_cuda=True

    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),])), batch_size=1, shuffle=True)

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Initialize the network
    # model = Net().to(device)
    model = MNIST_LeNet_plus(0.0, 0).to(device)

    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    #model = torch.load(pretrained_model, map_location=device)

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
        print('eps: ', eps, 'acc: ', acc)


