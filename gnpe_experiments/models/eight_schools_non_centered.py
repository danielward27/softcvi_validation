# %%
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution, Cauchy, Normal
from flowjax.experimental.numpyro import sample
from numpyro import plate

from gnpe_experiments.utils import MLPParameterizedDistribution


class EightSchoolsNonCenteredModel(eqx.Module):
    num_schools = 8
    sigma = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])
    y = jnp.array([28, 8, -3, 7, -1, 1, 18, 12])
    obs_name = "y"

    def __call__(self, obs=None):
        mu = sample("mu", Normal(0, 5))
        tau = sample("tau", Cauchy(0, 5))

        with plate("num_schools", self.num_schools):
            theta_trans = sample("theta_trans", Normal())
        sample("y", Normal(mu + tau * theta_trans, self.sigma), obs=obs)


class EightSchoolsNonCenteredGuide(eqx.Module):
    num_schools = 8
    theta_trans: AbstractDistribution
    mu: AbstractDistribution
    tau: AbstractDistribution

    def __init__(self, key, **kwargs):

        key, *keys = jr.split(key, 3)
        self.mu, self.tau = (
            MLPParameterizedDistribution(
                key,
                Normal(),
                cond_dim=self.num_schools,
                **kwargs,
            )
            for key in keys
        )

        self.theta_trans = MLPParameterizedDistribution(
            key,
            Normal(),
            cond_dim="scalar",
            **kwargs,
        )

    def __call__(self, obs):
        sample("mu", self.mu, condition=obs)
        sample("tau", self.tau, condition=obs)

        with plate("num_schools", self.num_schools):
            sample("theta_trans", self.theta_trans, condition=obs)
