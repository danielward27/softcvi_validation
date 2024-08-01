import jax.numpy as jnp
import pytest

from softcvi_validation.distributions import (
    PositiveImproperUniform,
    UniformWithLogisticBase,
)


def test_positive_improper_uniform():
    dist = PositiveImproperUniform()
    assert dist.log_prob(-10) == -jnp.inf
    assert pytest.approx(0) == dist.log_prob(5)
    assert pytest.approx(0) == dist.log_prob(100)


def test_uniform_with_logistic_base():
    uniform = UniformWithLogisticBase(23, 25)

    assert pytest.approx(jnp.log(0.5)) == uniform.log_prob(24)
    assert -jnp.inf == uniform.log_prob(22)
    assert -jnp.inf == uniform.log_prob(26)
