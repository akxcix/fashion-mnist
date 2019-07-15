import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class Fashion_MNIST_Data(object):
    def __init__(self,csv_path):
        """ Constructor """
        self.csv_frame = pd.read_csv(csv_path)

    def __len__(self):
        """ Returns Length """
        return self.csv_frame.shape[0]

    def return_label(self,idx):
        """ Returns one-hot vector of label """
        one_hot_dict = {
            0:np.array([1,0,0,0,0,0,0,0,0,0]),
            1:np.array([0,1,0,0,0,0,0,0,0,0]),
            2:np.array([0,0,1,0,0,0,0,0,0,0]),
            3:np.array([0,0,0,1,0,0,0,0,0,0]),
            4:np.array([0,0,0,0,1,0,0,0,0,0]),
            5:np.array([0,0,0,0,0,1,0,0,0,0]),
            6:np.array([0,0,0,0,0,0,1,0,0,0]),
            7:np.array([0,0,0,0,0,0,0,1,0,0]),
            8:np.array([0,0,0,0,0,0,0,0,1,0]),
            9:np.array([0,0,0,0,0,0,0,0,0,1])
        }
        # return torch.from_numpy(one_hot_dict[self.csv_frame.iloc[idx,0]].reshape([1,10])).type("torch.LongTensor")
        return torch.from_numpy(one_hot_dict[self.csv_frame.iloc[idx,0]]).type("torch.LongTensor")

    def __getitem__(self, idx):
        """ Returns tensor object with shape 1x28x28 """
        return torch.from_numpy(self.csv_frame.iloc[idx,1:].values.reshape([1,28,28])).type("torch.FloatTensor"), self.return_label(idx)

    def show_as_image(self,idx):
        """ Displays the image """
        tensor_object = torch.from_numpy(self.csv_frame.iloc[idx,1:].values.reshape([28,28]))
        np_image = tensor_object.numpy()
        plt.imshow(np_image)
        label = self.return_label(idx)
        print(label)        

# if __name__ == "__main__":
#     train_dataset = Fashion_MNIST_Data('./data/train.csv')
#     print(train_dataset[0][1].shape)
#     print(train_dataset.return_label(0))
#     train_dataset.show_as_image(0)
#     plt.show()