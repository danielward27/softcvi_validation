import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.distributions import Normal
from jax.scipy import integrate

from cnpe_validation.utils import Folded, TruncNormal


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


def test_truncnorm():
    tnorm = TruncNormal(lower=-2, upper=2, loc=1, scale=2)
    samp = tnorm.sample(jr.PRNGKey(0), (1000,))
    assert samp.max() < tnorm.upper
    assert samp.max() > tnorm.upper - 0.1

    assert samp.min() > tnorm.lower
    assert samp.min() < tnorm.lower + 0.1
