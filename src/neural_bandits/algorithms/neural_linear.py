# # train a neural network for representation learning and use a linear bandit algorithm to make decisions
# from .abstract_bandit import AbstractBandit

# from typing import List

# import torch
# import numpy as np
# import numpy.typing as npt


# class  NeuralLinearBandit(AbstractBandit):
#   def __init__(self, n_arms: int, n_features: int, n_hidden_units: int) -> None:
#     super().__init__(n_arms, n_features)
#     self.n_hidden_units = n_hidden_units
#     self.model = torch.nn.Sequential(
#       torch.nn.Linear(n_features, n_hidden_units),
#       torch.nn.ReLU(),
#       torch.nn.Linear(n_hidden_units, n_arms)
#     )
#     self.optimizer = torch.optim.Adam(self.model.parameters())

#   def predict(self, : List[npt.NDArray[np.int_]]) -> List[int]:
#     feature_vectors = self.model(torch.tensor(feature_vectors).float()).detach().numpy()
