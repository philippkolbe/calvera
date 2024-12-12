from .abstract_trainer import AbstractTrainer
from ..algorithms.linear_bandits import LinearBandit
import torch


class LinearTrainer(AbstractTrainer[LinearBandit]):
    def __init__(self) -> None:
        pass

    def update(
        self,
        bandit: LinearBandit,
        rewards: torch.Tensor,
        chosen_actions: torch.Tensor,
    ) -> LinearBandit:
        """Perform an update"""
        
        bandit.M += torch.einsum("bj,bk->jk", chosen_actions, chosen_actions)
        bandit.b += torch.einsum("b,bj->j", rewards, chosen_actions)
        bandit.theta = torch.inverse(bandit.M) @ bandit.b
        
        return bandit
        