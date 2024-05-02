import jax.numpy as jnp
import pytest
from flowjax.distributions import Normal
from jax.scipy import integrate

from cnpe_validation.utils import Folded


def test_folded():
    folded = Folded(Normal())

    assert pytest.approx(jnp.exp(folded.log_prob(1))) == 2 * jnp.exp(
        Normal().log_prob(1),
    )

    folded = Folded(Normal(-5, 2))
    x = jnp.linspace(-50, 50, num=1000)
    y = jnp.exp(folded.log_prob(x))
    integral = integrate.trapezoid(y, x)
    assert pytest.approx(1, abs=0.01) == integral
