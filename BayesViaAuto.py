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

    download=False,
)

loader = Data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)


class BayesAuto(nn.Module):
    def __init__(self):
        super(BayesAuto, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=4, stride=2, padding=4),
            nn.BatchNorm2d(20),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=1))
            #【1， 20， 17， 17】

        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 10, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stde=1))
            #【1， 10， 10， 10]
            #zeroing out individual concept feature images of the coded activations


        self.layer3 = nn.AvgPool2d(kernel_size=2, stride=2)
        #【1， 10， 5， 5】
        #.view(-1, 5*5*10)

        self.layer4 = nn.Sequential(
            nn.Linear(250, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        idx_feature = numpy.random.randint(0, 9, 1)
        x2[0, idx_feature, :, :] = 0 #zeroing out individual concept feature images of the coded activations
        x3 = self.layer3(x2)
        x3 = x3.view(-1, 5 * 5 * 10)
        x4 = self.layer4(x3)

        return x2, x3, x4

net = BayesAuto()

param_Autoen = torch.load('autoen.pkl')
#param_Bayes = torch.load('bayes.pkl')

param_model = net.state_dict()

#for i, j in param_Autoen.items():
    #for k, l in param_model.items():




#Test
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x, label) in enumerate(loader):

        x2, x3, x4 = net(x)
        loss = loss_func(x4, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss.data)

    print('________________________________________')
    print('finish training')





