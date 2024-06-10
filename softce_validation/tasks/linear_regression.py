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
from softce.models import AbstractGuide, AbstractModel

from softce_validation.tasks.tasks import AbstractTask


class LinearRegressionModel(AbstractModel):
    reparameterized: bool | None
    observed_names = {"y"}
    reparam_names = set()
    sigma: float | int
    n_covariates: ClassVar[int] = 20
    n_obs: ClassVar[int] = 200
    x: Float[Array, "200 20"]

    def __init__(self, key: PRNGKeyArray):
        self.x = jr.normal(key, (self.n_obs, self.n_covariates))
        self.reparameterized = None
        self.sigma = 1

    def call_without_reparam(
        self,
        obs: dict[str, Float[Array, " 200"]] | None = None,
    ):
        obs = obs["y"] if obs is not None else None

        beta = sample("beta", Normal(jnp.zeros(self.n_covariates)))
        bias = sample("bias", Normal())

        with plate("n_obs", self.n_obs):
            mu = self.x @ beta + bias
            sample("y", ndist.Normal(mu, self.sigma), obs=obs)

    def get_true_posterior(self, obs: dict):
        x = jnp.concatenate((jnp.ones((self.n_obs, 1)), self.x), axis=1)
        prior_means = jnp.zeros(x.shape[1])
        error_precision = 1 / self.sigma**2
        prior_precision = jnp.eye(x.shape[1])
        posterior_covariance = jnp.linalg.inv(
            error_precision * x.T @ x + prior_precision,
        )
        posterior_means = (
            posterior_covariance @ (error_precision * x.T @ obs["y"])
            + prior_precision @ prior_means
        )

        return {
            "beta": MultivariateNormal(
                posterior_means[1:],
                posterior_covariance[1:, 1:],
            ),
            "bias": Normal(posterior_means[0], posterior_covariance[0, 0] ** 0.5),
        }


class LinearRegressionGuide(AbstractGuide):
    beta: Normal
    bias: Normal

    def __init__(self):
        self.beta = Normal(jnp.zeros(LinearRegressionModel.n_covariates))
        self.bias = Normal()

    def __call__(self):
        sample("beta", self.beta)
        sample("bias", self.bias)


class LinearRegressionTask(AbstractTask):
    model: LinearRegressionModel
    guide: LinearRegressionGuide
    name = "linear_regression"

    def __init__(self, key: PRNGKeyArray):
        self.model = LinearRegressionModel(key)
        self.guide = LinearRegressionGuide()

    def get_latents_and_observed(
        self,
        key: Array,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        obs_key, posterior_key = jr.split(key)
        obs = self.model.reparam(set_val=False).sample(obs_key)
        obs = {k: obs[k] for k in self.model.observed_names}
        posterior = self.model.get_true_posterior(obs)

        keys = jr.split(posterior_key, len(posterior))
        latents = {
            name: dist.sample(key, (10000,))
            for (name, dist), key in zip(posterior.items(), keys, strict=True)
        }
        return latents, obs
