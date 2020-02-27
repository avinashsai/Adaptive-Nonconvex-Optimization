from __future__ import print_function
import os
import re
import sys
import random
import math
import string
import copy
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
seed = 1332
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from utils import *
import models
from models.resnet20 import *
from models.resnet50 import *
from train import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='yogi optimizer experiments')
    parser.add_argument('--task', type=str, default='image classification',
                        help='Experiment to run (default: image classification)')
    parser.add_argument('--nocuda', type=bool, default=False, 
                        help='Flag to disable cuda')
    parser.add_argument('--runs', type=int, default=6, 
                        help='No of runs (default: 6)')
    
    args = parser.parse_args()
    if(args.task=='image classification'):
        print("---------------------------------------------------------------------")
        print("Running Image Classification on CIFAR-10 Dataset")
        print("Loading CIFAR-10 Dataset")
        train_loader, test_loader = get_cifar10()
        print("Loaded Dataset")
        print("For running ResNet-20 enter 1")
        print("For running ResNet-50 enter 2")

        val = int(input())
        if(val !=1 and val != 2):
            raise ValueError("You have selected wrong choice {}. Please select 1 or 2".
                             format(val))
        if(not args.nocuda):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if(args.nocuda and device=='cuda'):
            warnings.warn('You have GPU but still preferred CPU. Training will be slow')
        elif(device=='cpu'):
            warning.warn("You don't have GPU. Training will be lot slower")
        
        avg_test_loss = []
        avg_test_accuracy = []
        for run in range(args.runs):
            if(val == 1):
                model = models.resnet20.ResNet20(BasicBlock20, [3, 3, 3]).to(device)
                print("-------------------------------------------------------------")
                print("Training ResNet20 on CIFAR-10")         
            else:
                print("-------------------------------------------------------------")
                print("Training ResNet50 on CIFAR-10")
                model = models.resnet50.ResNet50(BasicBlock50, [3, 4, 6, 3]).to(device)
        
            num_epochs = 1
            test_loss, test_acc = train(model, device, train_loader, test_loader, num_epochs, 
                                        val_loader=None)
            print("------|----------------|----------------|")
            print("  Run |    Test Loss   |  Test Accuracy  ")
            print("------|----------------|----------------|")
            print("  {}   |     {}      |        {}        ".format(run+1, test_loss, 
                                                                      test_acc))
                  
            avg_test_loss.append(test_loss)
            avg_test_accuracy.append(test_acc)
         
        avg_test_loss = np.asarray(avg_test_loss,'float32')
        avg_test_accuracy = np.asarray(avg_test_accuracy,'float32')
        print("-----------------------------------------------")
        print("Test Accuracy Mean {} STD {} ".format(round(np.mean(avg_test_accuracy),2),
                                                       round(np.std(avg_test_accuracy),2)))
     

if __name__ == '__main__':
    main()
