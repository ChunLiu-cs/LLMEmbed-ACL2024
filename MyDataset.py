import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self, mode, l_path, b_path, r_path):
        self.l_sents_reps = torch.load(l_path + f'{mode}_sents.pt')
        self.b_sents_reps = torch.load(b_path + f'{mode}_sents.pt')
        self.r_sents_reps = torch.load(r_path + f'{mode}_sents.pt')

        self.labels = torch.load(l_path + f'{mode}_labels.pt')
        
        self.sample_num = self.labels.shape[0]
        
    def __getitem__(self, index):
        return self.l_sents_reps[index], self.b_sents_reps[index], self.r_sents_reps[index], self.labels[index]        
        
    def __len__(self):
        return self.sample_num

