"""Linear regression task."""

from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as ndist
from flowjax.distributions import (
    MultivariateNormal,
    Normal,
)
from flowjax.experimental.numpyro import sample
from jaxtyping import Array, Float, PRNGKeyArray
from numpyro import plate
from paramax.wrappers import NonTrainable, non_trainable, unwrap
from pyrox.program import AbstractProgram

from softcvi_validation.tasks.tasks import AbstractTask


class LinearRegressionModel(AbstractProgram):
    """The model for the linear regression task.

    Args:
    key: The key used to generate the covariate data.
    """

    sigma: float | int
    n_covariates: ClassVar[int] = 50
    n_obs: ClassVar[int] = 200
    x: NonTrainable[Float[Array, "200 50"]]

    def __init__(self, key: PRNGKeyArray):
        x = jr.normal(key, (self.n_obs, self.n_covariates))
        self.x = non_trainable(x)
        self.sigma = 1

    def __call__(
        self,
        obs: Float[Array, " 200"] | None = None,
    ):
        self = unwrap(self)
        beta = sample("beta", Normal(jnp.zeros(self.n_covariates)))
        bias = sample("bias", Normal())

        with plate("n_obs", self.n_obs):
            mu = self.x @ beta + bias
            sample("y", ndist.Normal(mu, self.sigma), obs=obs)

    def get_true_posterior(self, obs: Array):
        self = unwrap(self)
        x = jnp.concatenate((jnp.ones((self.n_obs, 1)), self.x), axis=1)
        prior_means = jnp.zeros(x.shape[1])
        error_precision = 1 / self.sigma**2
        prior_precision = jnp.eye(x.shape[1])
        posterior_covariance = jnp.linalg.inv(
            error_precision * x.T @ x + prior_precision,
        )
        posterior_means = (
            posterior_covariance @ (error_precision * x.T @ obs)
            + prior_precision @ prior_means
        )

        return {
            "beta": MultivariateNormal(
                posterior_means[1:],
                posterior_covariance[1:, 1:],
            ),
            "bias": Normal(posterior_means[0], posterior_covariance[0, 0] ** 0.5),
        }


class LinearRegressionGuide(AbstractProgram):
    """Independent normal guide for the linear regression task."""

    beta: Normal
    bias: Normal

    def __init__(self):
        self.beta = Normal(jnp.zeros(LinearRegressionModel.n_covariates))
        self.bias = Normal()

    def __call__(self):
        sample("beta", self.beta)
        sample("bias", self.bias)


class LinearRegressionTask(AbstractTask):
    """A Bayesian linear regression task.

    Args:
        key: Jax random seed, used to generate toy covariate data.
    """

    model: AbstractProgram
    guide: LinearRegressionGuide
    name = "linear_regression"
    learning_rate = 2e-3
    observed_name = "y"
    latent_names = frozenset({"beta", "bias"})

    def __init__(self, key: PRNGKeyArray):
        self.model = LinearRegressionModel(key)
        self.guide = LinearRegressionGuide()

    def get_latents_and_observed(
        self,
        key: Array,
    ) -> tuple[dict[str, Array], Array]:
        obs_key, posterior_key = jr.split(key)
        obs = self.model.sample(obs_key)
        obs = obs.pop(self.observed_name)
        posterior = self.model.get_true_posterior(obs)
        keys = jr.split(posterior_key, len(posterior))
        latents = {
            name: dist.sample(key, (10000,))
            for (name, dist), key in zip(posterior.items(), keys, strict=True)
        }
        return latents, obs
