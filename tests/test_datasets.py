import pytest
import torch

from neural_bandits.datasets.covertype import CovertypeDataset
from neural_bandits.datasets.mnist import MNISTDataset
from neural_bandits.datasets.statlog import StatlogDataset
from neural_bandits.datasets.wheel import WheelBanditDataset


class TestCoverTypeDataset:
    @pytest.fixture
    def dataset(self) -> CovertypeDataset:
        return CovertypeDataset()

    def test_len(self, dataset: CovertypeDataset) -> None:
        assert len(dataset) == 581012

    def test_getitem(self, dataset: CovertypeDataset) -> None:
        for _ in range(50):
            X, y = dataset[0]
            assert X.shape == (54,)
            assert y.shape == (7,)
            assert torch.allclose(y.sum(), torch.tensor(1.0))


class TestMNISTDataset:
    @pytest.fixture
    def dataset(self) -> MNISTDataset:
        return MNISTDataset()

    def test_len(self, dataset: MNISTDataset) -> None:
        assert len(dataset) == 70000

    def test_getitem(self, dataset: MNISTDataset) -> None:
        for _ in range(50):
            X, y = dataset[0]
            assert X.shape == (784,)
            assert y.shape == (10,)
            assert torch.allclose(y.sum(), torch.tensor(1.0))


class TestStatlogDataset:
    @pytest.fixture
    def dataset(self) -> StatlogDataset:
        return StatlogDataset()

    def test_len(self, dataset: StatlogDataset) -> None:
        assert len(dataset) == 6435

    def test_getitem(self, dataset: StatlogDataset) -> None:
        for _ in range(50):
            X, y = dataset[0]
            assert X.shape == (36,)
            assert y.shape == (7,)
            assert torch.allclose(y.sum(), torch.tensor(1.0))
