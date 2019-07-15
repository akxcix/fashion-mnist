import torch
import torch.nn as nn
from dataset import FashionMnistDataset
from models import Model1
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.utils.data as D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.count = 0
        self.min_loss = np.Inf
        self.stop_training = False

    def save_checkpoint(self, model_state_dict):
        torch.save(model_state_dict, 'saved_weights/early_stopping_checkpoint.pth')


    def check(self, loss, model_state_dict):
        if loss < self.min_loss:
            self.min_loss = loss
            self.count = 0
        else:
            self.count +=1
            if self.count>self.patience:
                self.save_checkpoint(model_state_dict)
                self.stop_training = True
                print('EARLY STOPPING',self.count,self.min_loss,loss)



def save_model(model_state_dict,iter,path,train_acc,val_acc):
    """
    Save the model
    """
    torch.save(model_state_dict,path+'Model_epoch_no_'+str(iter)+'_train_acc_'+str(train_acc*100)[:5]+'_val_acc_'+str(val_acc*100)[:5]+'.pth')

# Set device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
TRAIN_VAL_RATIO = 0.75
print()
print("Device :",device)
print("Learning rate :",LEARNING_RATE)
print("Batch Size :",BATCH_SIZE)
print("Num of epochs :",EPOCHS)
print("Train-Val ratio :",TRAIN_VAL_RATIO)

# Dataloading
TRAIN_PATH = 'data/train.csv'
dataset = FashionMnistDataset(TRAIN_PATH)
train_len = int(TRAIN_VAL_RATIO*dataset.__len__())
val_len = dataset.__len__() - train_len
train_dataset, val_dataset = D.random_split(dataset,lengths=[train_len,val_len])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset,batch_size=64, shuffle=True,num_workers=4)

# Model Initialization
model = Model1()
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopper = EarlyStopping(3)

# Training
print("Starting Training...")
loss_arr = []
x_axis = []
x_accuracy_axis = []
train_accuracy_list = []
cross_val_accuracy_list = []
for epoch in tqdm(range(EPOCHS),desc="Epochs: ", ascii=False):
    
    if early_stopper.stop_training:
        break

    running_loss = 0.0
    for z in enumerate(tqdm(train_loader,desc="Batches: ",ascii=False)):
        if early_stopper.stop_training:
            break
        i, (images,labels) = z
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(images)
        loss = loss_fn(pred,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            tqdm.write('[%d, %5d] loss: %.3f'%(epoch+1,i+1, running_loss/100))
            loss_arr.append(running_loss/100)
            x_axis.append(epoch + i/len(train_loader))
            running_loss=0.0

    if epoch % 10 == 9:
        outputs_arr = []
        labels_arr = []
        with torch.no_grad():
            for (images,labels) in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = outputs.cpu().numpy()
                outputs = np.argmax(outputs, axis=1)
                labels = labels.cpu().numpy()
                outputs_arr= outputs_arr + list(outputs)
                labels_arr = labels_arr + list(labels)
            train_accuracy = accuracy_score(labels_arr,outputs_arr)
            tqdm.write("Training Accuracy   : "+str(train_accuracy))
            train_accuracy_list.append(train_accuracy)
            x_accuracy_axis.append(epoch+1)

        outputs_arr = []
        labels_arr = []
        with torch.no_grad():
            val_running_loss = 0.0
            for (images,labels) in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = loss_fn(outputs,labels)
                val_running_loss += val_loss.item()
                outputs = outputs.cpu().numpy()
                outputs = np.argmax(outputs, axis=1)
                labels = labels.cpu().numpy()
                outputs_arr= outputs_arr + list(outputs)
                labels_arr = labels_arr + list(labels)
            early_stopper.check(val_running_loss,model.state_dict())

            val_accuracy = accuracy_score(labels_arr,outputs_arr)
            tqdm.write("Validation Accuracy : "+str(val_accuracy))
            cross_val_accuracy_list.append(val_accuracy)
        save_model(model.state_dict(),epoch+1,'saved_weights/',train_accuracy_list[-1],cross_val_accuracy_list[-1])
print("\nTraining Done!")



plt.plot(x_axis,loss_arr,label='Train Loss')
plt.legend()
plt.show()

plt.plot(train_accuracy_list,label="Train Accuracy")
plt.plot(cross_val_accuracy_list, label = "Validation Accuracy")
plt.legend()
plt.show()

torch.save(model.state_dict(), 'saved_weights/model_state_dict.pth')
