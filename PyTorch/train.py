import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as D
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from yogi import *


def evaluate(net, loader, device):
    net.eval()
    with torch.no_grad():
        testloss = 0.0
        testacc = 0.0
        total = 0.0
        for indices, labels in loader:
            indices, labels = indices.to(device), labels.to(device)
            total += indices.size(0)
            output = net(indices)
            loss = F.cross_entropy(output, labels, reduction='sum')
            testloss += loss.item()
            output = torch.argmax(output,1)
            testacc += (output==labels).sum().item()

        return round((testloss / total), 3), round((testacc / total) * 100, 3)

def train(model, device, train_loader, test_loader, numepochs, val_loader):
    """Model Training

    Arguments:
        model (PyTorch Model): Model to train
        device (string): Whether to train on GPU or CPU
        train_loader (DataLoader): Training Data Loader
        test_loader (DataLoader): Testing Data Loader
        numepochs (int): Number of epochs to train the model
        val_loader (DataLoader, optional): Validation Data Loader (default: None)

    """
    yogi = Yogi(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-3, 
                initial_accumulator=1e-6)
    
    scheduler_yogi = optim.lr_scheduler.ReduceLROnPlateau(yogi, mode='min', 
                                                          factor=0.5, patience=20)
    
    numbatches = len(train_loader)
    model.train()
    for epoch in range(1,numepochs+1):
        model.train()
        curtrainloss = 0.0
        for trainind,trainlab in train_loader:
            yogi.zero_grad()
            trainind, trainlab = trainind.to(device), trainlab.to(device)

            output = model(trainind)
            loss = F.cross_entropy(output, trainlab)
            curtrainloss+=loss.item()

            loss.backward()
            yogi.step()

        curtrainloss_ = curtrainloss/numbatches
        scheduler_yogi.step(curtrainloss)
        if(epoch%25==0):
            print("Epoch {} Training Loss {} ".format(epoch, curtrainloss))

    test_loss, test_acc = evaluate(model, test_loader, device)
    return test_loss, test_acc
