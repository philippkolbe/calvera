from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt


class AbstractBandit(ABC):
    def __init__(self, n_arms: int, n_features: int) -> None:
        self.n_arms = n_arms
        self.n_features = n_features

    @abstractmethod
    def predict(self, feature_vectors: List[npt.NDArray[np.int_]]) -> List[int]:
        """Predict a list of multiple sets of contextualised actions"""
        pass

    @abstractmethod
    def select_arm(self, contextualised_actions: npt.NDArray[np.int_]) -> int:
        """Select an arm based on the contextualised actions"""
        pass

    @abstractmethod
    def update_step(
        self, reward: float, chosen_contextualised_action: npt.NDArray[np.int_]
    ) -> None:
        """Perform a single update step"""
        pass

    @abstractmethod
    def update_batch(
        self,
        rewards: List[float],
        chosen_contextualised_actions: List[npt.NDArray[np.int_]],
    ) -> None:
        """Perform a batch update"""
        pass
