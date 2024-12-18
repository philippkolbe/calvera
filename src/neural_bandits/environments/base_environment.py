import random
from typing import Any, Callable, Dict, Generic, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..trainers.abstract_trainer import AbstractTrainer, BanditType


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class BaseEnvironment(Generic[BanditType]):
    """
    A simple environment for training and evaluating bandit algorithms
    on a given dataset/scenario. The environment takes in a bandit instance,
    a dataset, and some optional parameters (like random seeds for reproducibility).
    """

    def __init__(
        self,
        bandit: BanditType,
        trainer: AbstractTrainer[BanditType],
        dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]],
        reward_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        seed: int = 42,
        max_steps: int | None = None,
        log_method: Callable[[Dict[str, Any]], None] = lambda x: None,
    ) -> None:
        self.bandit = bandit
        self.dataset = dataset
        self.trainer = trainer
        self.reward_fn = reward_fn
        self.log = log_method
        self.max_steps = max_steps or len(dataset)  # type: ignore
        self.seed = seed

    def run(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Run the bandit algorithm on the given dataset scenario and record metrics.
        """
        seed_all(self.seed)
        indices = np.arange(len(self.dataset))  # type: ignore
        np.random.shuffle(indices)

        chosen_arms = []
        result_rewards = []
        cumulative_rewards = []
        cum_reward = 0.0

        for t in range(0, self.max_steps, batch_size):
            context_list, correct_arm_list = [], []

            for idx in indices[t : t + batch_size]:
                x, y = self.dataset[idx]
                context_list.append(x)
                correct_arm_list.append(y)

            contexts = torch.stack(context_list)
            correct_arms = torch.stack(correct_arm_list)

            arms = torch.argmax(self.bandit(contexts), dim=-1)
            chosen_arms.append(arms)

            rewards = self.reward_fn(arms, correct_arms, contexts)
            result_rewards.append(rewards)
            self.bandit = self.trainer.update(
                self.bandit, rewards=rewards, chosen_actions=arms
            )

            self.log(
                {
                    "t": t,
                    "arm": arms,
                    "reward": rewards,
                    "cumulative_reward": cum_reward + rewards,
                }
            )

            for i in range(len(rewards)):
                cum_reward += rewards[i].item()
                self.log(
                    {
                        "step": t + i,
                        "reward": rewards[i].item(),
                        "cumulative_reward": cum_reward,
                    }
                )
                cumulative_rewards.append(cum_reward)

        results = {
            "chosen_arms": torch.cat(chosen_arms).tolist(),
            "rewards": torch.cat(result_rewards).tolist(),
            "cumulative_rewards": cumulative_rewards,
            "final_cumulative_reward": cum_reward,
        }
        return results


# Example usage:
# TODO: Add example usage
