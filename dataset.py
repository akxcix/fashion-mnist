#!/usr/bin/env python3

# Author: Adarsh Kumar
# GitHub/GitLab: @iamadarshk

import torch
import numpy as np
import pandas as pd

class FashionMnistDataset(object):
    '''
    Representation of the Dataset
    '''

    def __init__(self, csv_path):
        """ Constructor"""
        self.csv_path = csv_path
        self.dframe = pd.read_csv(csv_path)
    
    def __len__(self):
        """ returns length of dataset"""
        return self.dframe.shape[0]

    def return_label(self,idx):
        "return label"
        return torch.tensor(self.dframe.iloc[idx,0])

    def __getitem__(self,idx):
        "returns image as torch.FloatTensor of shape [1 x 28 x 28] and label as torch.LongTensor in a tuple as (image,label)"
        return ((torch.from_numpy(((self.dframe.iloc[idx,1:].values)/255).reshape([1,28,28]))).type("torch.FloatTensor"), self.return_label(idx))