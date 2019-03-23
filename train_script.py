import numpy as np
import pandas as pd
from datasets import Fashion_MNIST_Data
from models import CNNModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# to set GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


######################## Hyperparameters ###########################################
learning_rate = 1e-3
batch_size = 64
num_epochs = 100
####################################################################################


####################### Dataloading ################################################
train_dataset = Fashion_MNIST_Data("./data/train.csv")
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = Fashion_MNIST_Data("./data/test.csv")
testloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=4)
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
####################################################################################


####################### Network ####################################################
net = CNNModel()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)
####################################################################################


####################### Training ###################################################
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, torch.max(labels,1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f'%(epoch+1, i+1, running_loss/100))
            running_loss = 0.0
    
    if epoch % 20 == 19:
        correct = 0
        total=0
        with torch.no_grad():
            for (images,labels) in trainloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.max(labels,1)[1]).sum().item()

        print('Accuracy of the network on the train images: %d %%' % (
            100 * correct / total))

print('Finished Training')
####################################################################################


###################### Testing #####################################################
correct = 0
total = 0
with torch.no_grad():
    for (images,labels) in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == torch.max(labels,1)[1]).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
#####################################################################################


