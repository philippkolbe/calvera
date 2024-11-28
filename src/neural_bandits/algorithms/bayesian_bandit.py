import numpy as np


class BayesianBandit:
    def __init__(self, n_arms, n_features):
        self.M = np.stack([np.eye(n_features) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, n_features))  # Reward vector
        self.theta = np.random.uniform(
            -1, 1, (n_arms, n_features)
        )  # Parameter estimate
        self.arms = n_arms
        self.gen = np.random.default_rng()

    def predict(self, feature_vectors):
        results = []
        for i in range(len(feature_vectors)):
            results.append(self.select_arm(feature_vectors[i]))
        return results

    def select_arm(self, x):
        p_values = []
        for k in range(self.arms):
            theta = self.gen.multivariate_normal(
                mean=self.theta[k],
                cov=np.linalg.inv(self.M[k]),
            )
            p_values.append(x.dot(theta))
        return np.argmax(p_values)

    def update(self, chosen_arm, reward, x):

        self.M[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x
        self.theta[chosen_arm] = np.linalg.inv(self.M[chosen_arm]).dot(
            self.b[chosen_arm]
        )