import pickle as pkl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class advDataset(Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, 'rb') as fp:
            self.data = pkl.load(fp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][0].squeeze(0)).type(torch.float32)
        y = torch.tensor(self.data[idx][1]).type(torch.int64)

        return x, y 

if __name__ == '__main__':
    dset = advDataset('./adv_testset.pkl')

    dataloader = DataLoader(dset, batch_size=32)

    for x, y in dataloader:
        import pdb; pdb.set_trace()
