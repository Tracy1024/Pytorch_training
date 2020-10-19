import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy

EPOCH = 1
BATCH_SIZE = 1
LR = 0.001

data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),

    download=True,
)

loader = Data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)


for step, (x, target) in enumerate(loader):
    if(step == 2):
        print(target)


