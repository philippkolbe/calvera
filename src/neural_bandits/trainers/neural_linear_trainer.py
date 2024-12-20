import torch

from trainers.abstract_trainer import AbstractTrainer
from neural_bandits.algorithms.neural_linear import NeuralLinearBandit


class NeuralLinearTrainer(AbstractTrainer[NeuralLinearBandit]):
    def __init__(self) -> None:
        pass

    def update(
        self,
        bandit: NeuralLinearBandit,
        rewards: torch.Tensor,  # shape: (batch_size,)
        chosen_actions: torch.Tensor,  # shape: (batch_size, n_arms, n_features)
    ) -> NeuralLinearBandit:
        """Perform an update"""

        return bandit

    def update_nn(self):
        pass
