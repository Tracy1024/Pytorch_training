#from __future__ import print_function, division
import os
import torch
#import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms, utils
import matplotlib.image as mping
import torch.utils.data as Data
import torchvision
import os




EPOCH = 1
BATCH_SIZE = 1
LR = 0.001
DOWNLOAD_MNIST = False
train = False


class CNNauto(nn.Module):
    def __init__(self):
        super(CNNauto, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=4, stride=2, padding=4),
            nn.BatchNorm2d(20),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=1))

        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 10, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=1))

        self.layer3 = nn.Sequential(
            #nn.MaxUnpool2d(kernel_size=2, stride=1),
            nn.ConvTranspose2d(10, 20, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            #nn.MaxUnpool2d(kernel_size=2, stride=1),
            nn.ConvTranspose2d(20, 3, kernel_size=4, stride=2, padding=4),
            nn.Sigmoid())

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        r2 = self.layer3(f2)
        r1 = self.layer4(r2)
        return f1, f2, r2, r1
autoen = CNNauto()

optimizer = torch.optim.Adam(autoen.parameters(), lr=LR)
loss_func = nn.MSELoss()

root = '/mrtstorage/datasets/kitti/data/kitti_raw/2011_09_26/2011_09_26_drive_0091_extract/image_02/data'
files = os.listdir(root)
data_train = []
for image in files:
    img = mping.imread(root+"/"+image)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    img = torch.unsqueeze(img, 0)
    data_train.append(img)



if(train == True):
    for epoch in range(EPOCH):
        for x in data_train:

            f1, f2, r2, r1 = autoen(x)

            print(f1.size(), f2.size(), r2.size(), r1.size())

            loss = loss_func(r1, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss.data)

        print('________________________________________')
        print('finish training')
    torch.save(autoen.state_dict(), 'autoen_KIIT.pkl')


if(train == False):
    feature = []
    autoen.load_state_dict(torch.load('autoen_KIIT.pkl'))
    f1, f2, r2, r1 = autoen(data_train[20])
    feature = [f1, f2, r1, r2]

    r1 = r1.permute(0, 2, 3, 1)
    r1 = torch.squeeze(r1, 0)
    r1_ = r1.numpy()
    plt.plot(r1_)






