from abc import ABC, abstractmethod

import torch


class AbstractContextualiser(ABC): # e.g. disjoint model 
    @abstractmethod
    def contextualise(
        self, feature_vector: torch.Tensor
    ) -> torch.Tensor:
        pass
