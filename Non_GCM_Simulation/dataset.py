import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

class NoisySet(Dataset):
    def __init__(self, file_name, sensitive, args):
        self.target = "ZFYA"
        self.sensitive = sensitive
        self.df = pd.read_csv(file_name)
        self.data_flag = file_name.split("/")[-1].split("_")[1]
        
        self.make_env(args)
    
    def make_env(self, args):
        self.y = np.array(self.df[self.target]).reshape(-1, 1)
        self.s = np.array(self.df[self.sensitive]).reshape(-1, 1)
        self.X = np.array(self.df.drop([self.target, self.sensitive], axis=1))
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.s[idx]), \
            torch.FloatTensor(self.y[idx])
    
    def __len__(self):
        return len(self.X)

def get_xsy_loaders(train_file_name, test_file_name, sensitive, test_batch_size, args):
    trainset, testset = NoisySet(train_file_name, sensitive, args), \
        NoisySet(test_file_name, sensitive, args)
    train_batch_size = trainset.y.shape[0]
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, 
                              drop_last=False, num_workers=0)
    test_batch_size = testset.y.shape[0]
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, 
                             drop_last=False, num_workers=0)
    return train_loader, test_loader