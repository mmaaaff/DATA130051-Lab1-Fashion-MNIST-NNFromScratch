from MyDL.tensor import *
import numpy as np

class Dataset():
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def Dataloader(dataset, batch_size, shuffle=True):
    if isinstance(dataset, MyTensor):
        raise(TypeError("Dataset must be an instance of MyDL.data.Dataset instead of MyDL.tensor.MyTensor."))
    n = len(dataset)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    j = 0
    for i in range(0, n - batch_size + 1, batch_size):
        j = i + batch_size
        batch_indices = indices[i:j]
        yield dataset[batch_indices]
    if j < n:
        batch_indices = indices[j:]
        yield dataset[batch_indices]