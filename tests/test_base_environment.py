import pytest

from neural_bandits.environments.base_environment import BaseEnvironment


class TestBaseEnvironment: # TODO
    @pytest.fixture(
        params=[
            {
                "bandit": "bandit",
                "trainer": "trainer",
                "dataset": "dataset",
                "contextualiser": "contextualiser",
                "seed": 42,
                "max_steps": 100,
                "log_method": lambda x: None,
            },
        ]
    )
    def test_set(request) -> None:
        return BaseEnvironment(**request.param)
