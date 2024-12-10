import torch

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
        self,
        feature_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Performs the disjoint model contextualisation

        Args:
            feature_vector (torch.Tensor): Input feature vector of shape (batch_size, n_features)

        Returns:
            torch.Tensor: Contextualised actions of shape (bathch_size, n_arms, n_classes * n_arms)
        """
        # contextualised_actions = torch.outer(
        #     torch.eye(self.n_arms), feature_vector
        # ).reshape(self.n_arms, self.n_classes * self.n_arms)
        contextualised_actions = torch.einsum(
            "i,kj->k,ij", torch.eye(self.n_arms), feature_vector
        )

        return contextualised_actions
