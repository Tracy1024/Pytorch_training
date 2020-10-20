import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 1
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),

    download=False,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class CNNauto(nn.Module):
    def __init__(self):
        super(CNNauto, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=4, stride=2, padding=4),
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
            nn.ConvTranspose2d(20, 1, kernel_size=4, stride=2, padding=4),
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

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):

        print(x.size())
        f1, f2, r2, r1 = autoen(x)

        print(f1.size(), f2.size(), r2.size(), r1.size())

        loss = loss_func(r1, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss.data)

    print('________________________________________')
    print('finish training')
torch.save(autoen.state_dict(), 'autoen.pkl')

