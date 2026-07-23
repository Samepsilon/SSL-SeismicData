from torch.utils.data import Dataset
import torch

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data, transform=None):
        assert all(data[0].shape[0] == item.shape[0] for item in data)
        self.tensors = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return torch.tensor(x).float(), torch.tensor(y)

    def __len__(self):
        return self.tensors[0].shape[0]