from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import (
    Scale,
)
from flowjax.distributions import (
    AbstractDistribution,
    AbstractTransformed,
    Normal,
)
from flowjax.experimental.numpyro import sample
from jax import Array
from jax.scipy.stats import norm
from jax.typing import ArrayLike
from numpyro import deterministic, plate
from numpyro.distributions import HalfCauchy

from cnpe_validation.tasks.tasks import AbstractPosteriorDBTask
from cnpe_validation.utils import MLPParameterizedDistribution


class _StandardHalfNormal(AbstractDistribution):  # TODO test
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jnp.where(x > 0, norm.logpdf(x) + jnp.log(2), -jnp.inf)

    def _sample(self, key, condition=None):
        return jnp.abs(jr.normal(key, shape=self.shape))


class HalfNormal(AbstractTransformed):
    """Half normal distribution.

    Args:
        scale: The scale of the half normal distribution.
    """

    base_dist: _StandardHalfNormal
    bijection: Scale

    def __init__(self, scale: ArrayLike = 1):
        self.base_dist = _StandardHalfNormal(jnp.shape(scale))
        self.bijection = Scale(scale)


class EightSchoolsNonCenteredModel(eqx.Module):
    num_schools: ClassVar[int] = 8
    sigma: ClassVar[Array] = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])
    y = jnp.array([28, 8, -3, 7, -1, 1, 18, 12])

    def __call__(self, obs=None):
        mu = sample("mu", Normal(0, 5))
        tau = sample("tau", HalfCauchy(scale=5))

        with plate("num_schools", self.num_schools):
            theta_trans = sample("theta_trans", Normal())
        sample("y", Normal(mu + tau * theta_trans, self.sigma), obs=obs)


class EightSchoolsNonCenteredGuide(eqx.Module):
    theta_trans: AbstractDistribution
    mu: AbstractDistribution
    tau: AbstractDistribution

    def __init__(self, key, **kwargs):

        key, subkey = jr.split(key)
        self.mu = MLPParameterizedDistribution(
            subkey,
            Normal(),
            cond_dim=EightSchoolsNonCenteredModel.num_schools,
            **kwargs,
        )

        key, subkey = jr.split(key)
        self.tau = MLPParameterizedDistribution(
            subkey,
            HalfNormal(),
            cond_dim=EightSchoolsNonCenteredModel.num_schools,
            **kwargs,
        )

        key, subkey = jr.split(key)
        self.theta_trans = MLPParameterizedDistribution(
            subkey,
            Normal(),
            cond_dim="scalar",
            **kwargs,
        )

    def __call__(self, obs):
        mu = sample("mu", self.mu, condition=obs)
        tau = sample("tau", self.tau, condition=obs)

        with plate("num_schools", self.num_schools):
            theta_trans = sample("theta_trans", self.theta_trans, condition=obs)
            deterministic("theta", mu + tau * theta_trans)


class EightSchoolsNonCenteredTask(AbstractPosteriorDBTask):
    model: ClassVar[str] = EightSchoolsNonCenteredModel()
    guide: Callable
    obs_name: ClassVar[str] = "y"

    def __init__(self, key):
        self.guide = EightSchoolsNonCenteredGuide(key, width_size=20)
