import numpy as np
import pytest

from vh import SMPLModel


@pytest.fixture
def model() -> SMPLModel:
    """A small random model with the same structure as SMPL (not a body)."""
    rng = np.random.default_rng(0)
    V, K, B = 60, 24, 10
    parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
    weights = rng.random((V, K)) ** 4
    J_regressor = rng.random((K, V))
    return SMPLModel(
        v_template=rng.normal(size=(V, 3)),
        shapedirs=rng.normal(size=(V, 3, B)) * 0.05,
        posedirs=rng.normal(size=(V, 3, 9 * (K - 1))) * 0.01,
        J_regressor=J_regressor / J_regressor.sum(axis=1, keepdims=True),
        weights=weights / weights.sum(axis=1, keepdims=True),
        parents=parents,
        faces=rng.integers(0, V, size=(100, 3)),
    )
