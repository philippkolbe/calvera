from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt


class AbstractContextualiser(ABC):
    @abstractmethod
    def contextualise(
        self, feature_vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def contextualise_batch(
        self, feature_vectors: List[npt.NDArray[np.float64]]
    ) -> List[npt.NDArray[np.float64]]:
        pass
