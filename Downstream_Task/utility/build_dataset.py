import torch
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    """
    Dataset that applies two independent augmentation pipelines to the same
    sample, returning (x_i, x_j, y).

    Args:
        data          : tuple (X, y) where X is [N, C, L] and y is [N]
        transform_A   : callable applied to x to produce x_i  (default: identity)
        transform_B   : callable applied to x to produce x_j  (default: identity)
    """

    def __init__(self, data, transform_A=None, transform_B=None):
        assert all(data[0].shape[0] == item.shape[0] for item in data), \
            "X and y must have the same number of samples"
        self.X           = data[0]
        self.y           = data[1]
        self.transform_A = transform_A
        self.transform_B = transform_B

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]

        x_i = self.transform_A(x) if self.transform_A else x
        x_j = self.transform_B(x) if self.transform_B else x

        return (
            torch.tensor(x_i, dtype=torch.float32),
            torch.tensor(x_j, dtype=torch.float32),
            torch.tensor(self.y[idx]),
        )