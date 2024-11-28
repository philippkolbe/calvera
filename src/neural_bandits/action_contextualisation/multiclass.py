from typing import List

import numpy as np
import numpy.typing as npt

from .abtract_contextualiser import AbstractContextualiser


class MultiClassContextualiser(AbstractContextualiser):
    def __init__(
        self,
        n_arms: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.n_arms = n_arms
        self.n_classes = n_classes

    def contextualise(
        self, feature_vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        contextualised_actions = np.outer(
            np.identity(self.n_arms), feature_vector
        ).reshape(self.n_arms, self.n_classes * self.n_arms)

        return contextualised_actions

    def contextualise_batch(
        self, feature_vectors: List[npt.NDArray[np.float64]]
    ) -> List[npt.NDArray[np.float64]]:
        batched_actions = [
            self.contextualise(feature_vector) for feature_vector in feature_vectors
        ]

        return batched_actions
