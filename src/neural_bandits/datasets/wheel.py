from typing import Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray


class WheelBanditDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Generates a dataset for the Wheel Bandit problem.

    Args:
        num_samples (int): Number of samples to generate
        delta (float): The difference in reward for the two different regions
        radius (float): The radius of the inner circle
        seed (int): Seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int,
        delta: float = 0.5,
        radius: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.delta = delta
        self.radius = radius
        self.seed = seed
        self.data = self._generate_data()

    def _generate_data(self) -> list[Tuple[NDArray[np.float64], float]]:
        if self.seed is not None:
            np.random.seed(self.seed)

        data = []
        for _ in range(self.num_samples):
            x = np.random.uniform(-1, 1, size=2)
            if np.linalg.norm(x) < self.radius:
                reward = np.random.normal(1, 1)
            else:
                angle = np.arctan2(x[1], x[0])
                if angle < -np.pi / 4 or angle > np.pi / 4:
                    reward = np.random.normal(1 + self.delta, 1)
                else:
                    reward = np.random.normal(1 - self.delta, 1)
            data.append((x, reward))
        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, reward = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            reward, dtype=torch.float32
        )
