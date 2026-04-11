import torch
import numpy
from torch.utils.data import Dataset


class TraditionalDataset(Dataset):
    def __init__(self, texts, targets, max_len, hidden_size):
        self.texts = texts
        self.targets = targets
        self.max_len = max_len
        self.hidden_size = hidden_size
     

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        feature = self.texts[idx]
        target = self.targets[idx]
        vectors = numpy.zeros(shape=(3, self.max_len, self.hidden_size))
        for j in range(3):
            for i in range(min(len(feature[0]), self.max_len)):
                vectors[j][i] = feature[j][i]
        return {
            'vector': vectors,
            'targets': torch.tensor(target, dtype=torch.long)
        }
