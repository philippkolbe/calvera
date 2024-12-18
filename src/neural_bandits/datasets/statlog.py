from typing import Tuple

import torch
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo


class StatlogDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
