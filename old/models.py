import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # shape: 1x28x28
        self.conv1 = nn.Conv2d(1,3,5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=4,stride=1)
        # shape: 3x21x21
        self.conv2 = nn.Conv2d(3,6,5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=4,stride=1)
        # shape: 6x14x14
        self.conv3 = nn.Conv2d(6,12,5)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=4,stride=2)
        # shape: 12x7x7

        # flatten

        # shape: 1x588
        self.linear4 = nn.Linear(192,100)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)
        # shape: 1x250
        self.linear5 = nn.Linear(100,50)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.5)
        # shape: 1x100
        self.linear6 = nn.Linear(50,10)
        # shape:1x10
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.reshape(x.size(0),-1)
        x = self.linear4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.linear5(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        x = self.linear6(x)
        x = self.softmax(x)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out