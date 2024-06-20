import torch
import io
from torch.utils.data import Dataset

from random import randrange

class TextDataset(Dataset):
    def __init__(self, src, encoder, blockSize) -> None:
        super().__init__()
        self.encoder   = encoder
        self.blockSize = blockSize
        self.text      = ""

        f = io.open(src, mode="r", encoding="utf-8")
        self.text = f.read()
        f.close()

        self.data = torch.tensor(self.encoder.encode(self.text), dtype=torch.long)

    def __getitem__(self, index):
        index = index % self.__len__()
        r = randrange(0,self.blockSize) + (index * self.blockSize * 2)
        sample  = self.data[r     : r + self.blockSize]
        targets = self.data[r + 1 : r + self.blockSize + 1]
        return sample, targets
    
    def __len__(self):
        return self.data.shape[0] // (self.blockSize * 2)

