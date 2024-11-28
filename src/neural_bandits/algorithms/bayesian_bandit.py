from typing import List

import numpy as np
import numpy.typing as npt

from .abstract_bandit import AbstractBandit


class LinearBayesianBandit(AbstractBandit):
    def __init__(self, n_arms: int, n_features: int) -> None:
        super().__init__(n_arms, n_features)

        self.M: npt.NDArray[np.float64] = np.identity(n_features)
        self.b = np.zeros((n_features))
        self.theta = np.random.uniform(-1, 1, (n_features))
        self.gen = np.random.default_rng()

    def predict(self, feature_vectors: List[npt.NDArray[np.int_]]) -> List[int]:
        results = []
        for i in range(len(feature_vectors)):
            results.append(self.select_arm(feature_vectors[i]))
        return results

    def select_arm(self, contextualised_actions: npt.NDArray[np.int_]) -> int:
        theta_tilde = self.gen.multivariate_normal(
            self.theta, np.linalg.inv(self.M)
        )  # shape: (n_arms * n_features)

        # vectorised implementation
        return int(
            np.argmax(
                np.sum(
                    np.reshape(contextualised_actions, (self.n_arms, self.n_features))
                    * theta_tilde,
                    axis=1,
                )
            )
        )

    def update_step(
        self, reward: float, chosen_contextualised_action: npt.NDArray[np.int_]
    ) -> None:
        self.M += np.outer(chosen_contextualised_action, chosen_contextualised_action)
        self.b += reward * chosen_contextualised_action
        self.theta = np.linalg.inv(self.M).dot(self.b)

    def update_batch(
        self,
        rewards: List[float],
        chosen_contextualised_actions: List[npt.NDArray[np.int_]],
    ) -> None:
        self.M += np.sum(
            [
                np.outer(
                    chosen_contextualised_actions[i], chosen_contextualised_actions[i]
                )
                for i in range(len(rewards))
            ],
            axis=0,
        )
        self.b += np.sum(
            [
                rewards[i] * chosen_contextualised_actions[i]
                for i in range(len(rewards))
            ],
            axis=0,
        )
        self.theta = np.linalg.inv(self.M).dot(self.b)
