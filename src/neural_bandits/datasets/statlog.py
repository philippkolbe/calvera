from typing import Tuple

import torch
from ucimlrepo import fetch_ucirepo

from .base_dataset import AbstractDataset


class StatlogDataset(AbstractDataset):
    def __init__(self) -> None:
        dataset = fetch_ucirepo(id=146)
        X = dataset.data.features
        y = dataset.data.targets

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.y = (
            torch.nn.functional.one_hot(self.y - 1, num_classes=7).float().squeeze(1)
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]

    def reward(self, idx: int, action: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float(self.y[idx] == action + 1), dtype=torch.float32)
