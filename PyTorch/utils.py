import torch
import torchvision
from torchvision import datasets, transforms
datapath = '/CIFAR-10/'
train_batch_size = 128
test_batch_size = 64


def get_cifar10():
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainset = torchvision.datasets.CIFAR10(root=datapath, train=True,
          download=True, transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(),
                                            transform
                                            ]))
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=train_batch_size,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=datapath, train=False,
                                            download=True,
                                            transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transform
                                            ]))
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=test_batch_size,
                                             shuffle=False)
    return train_loader, test_loader
