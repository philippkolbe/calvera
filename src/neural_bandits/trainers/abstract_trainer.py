from ..algorithms.abstract_bandit import AbstractBandit

from abc import ABC, abstractmethod

import torch


class AbstractTrainer(ABC):
    @abstractmethod
    def update(
        self, bandit: AbstractBandit, rewards: torch.Tensor, chosen_actions: torch.Tensor,
    ) -> AbstractBandit:
        """Perform a single update step"""
        # TODO(rob2u): assert correct shapes (rewards: (batch_size, 1), chosen_actions: (batch_size, dim))
        
    
