import pandas as pd
import torch
import sys
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    # data is a dataframe; parsing happens elsewhere since we want to split the target training data for validation
    # label_key is the non-dummied column key of the label
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return self.features.shape[0]


    def __getitem__(self, index):
        row = self.features.iloc[index]
        target = self.labels.iloc[index]
        return (torch.tensor(row, dtype=torch.float32), torch.tensor(target, dtype=torch.int64)) # CPU tensors


    def __Nfeatures__(self):
        return self.features.shape[1]


    def __Nlabels__(self):
        return len(Counter(self.labels).keys())