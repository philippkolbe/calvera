from numpy.typing import NDArray
import numpy as np
from sklearn.utils import Bunch
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo
import torch
from typing import Tuple


class StatlogDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Loads the Covertype dataset as a pytorch Dataset from the UCI repository (https://archive.ics.uci.edu/dataset/98/statlog+project).

    Args:
        root (str): Where to store the dataset
        download (bool): Whether to download the dataset
    """

    data: Bunch
    X: NDArray[np.float32]
    y: NDArray[np.int64]

    def __init__(self, root: str = "./data", download: bool = True):
        self.data = fetch_ucirepo(id=144)
        self.X = self.data.data.features.astype(np.float32)
        self.y = self.data.data.targets.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X_item = torch.tensor(self.X[idx], dtype=torch.float32)
        y_item = torch.tensor(self.y[idx], dtype=torch.int64)
        return X_item, y_item
