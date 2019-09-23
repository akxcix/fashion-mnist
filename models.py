import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self):
        super(Model1,self).__init__()

        # 1 x 28 x 28
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5)
        self.relu1 = nn.ReLU()
        # 16 x 24 x 24
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # 16 x 12 x 12
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=5)
        self.relu2 = nn.ReLU()
        # 32 x 8 x 8
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # 32 x 4 x 4
        self.dense1 = nn.Linear(32 * 4 * 4, 10)
        # 10

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.dense1(x)
        return x

