from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_covtype
from sklearn.utils import Bunch
from torch.utils.data import Dataset
import torch


class CovertypeDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Loads the Covertype dataset as a pytorch Dataset from the UCI repository (https://archive.ics.uci.edu/ml/datasets/covertype).

    Args:
        root (str): Where to store the dataset
        download (bool): Whether to download the dataset
    """

    data: Bunch
    X: NDArray[np.float32]
    y: NDArray[np.int64]

    def __init__(self, root: str = "./data", download: bool = True):
        self.data = fetch_covtype(root=root, download=download)
        self.X = self.data.data.astype(np.float32)
        self.y = self.data.target.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.tensor(self.y[idx], dtype=torch.int64)
        return X_item, y_item
