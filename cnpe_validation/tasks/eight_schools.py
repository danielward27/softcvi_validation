from typing import ClassVar

import jax.numpy as jnp
import jax.random as jr
from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel
from cnpe.numpyro_utils import get_sample_site_names
from flowjax.bijections import Scale
from flowjax.distributions import (
    AbstractDistribution,
    AbstractTransformed,
    Normal,
)
from flowjax.experimental.numpyro import sample
from jax import Array
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray
from numpyro import plate
from numpyro.distributions import HalfCauchy

from cnpe_validation.tasks.tasks import (
    AbstractTaskWithReference,
    get_posterior_db_reference_posterior,
)
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


# TODO seperation of observation and other data seems a little strange, but perhaps
# logical since we simulate obs too.
class EightSchoolsModel(AbstractNumpyroModel):
    """Eight schools model. We reparamerterize (non-centering)."""

    obs_names = ("y",)
    reparam_names = ("mu", "theta")
    num_schools: ClassVar[int] = 8
    sigma: ClassVar[Array] = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])

    def call_without_reparam(
        self,
        obs: dict[str, Float[Array, " 8"]] | None = None,
    ):
        mu = sample("mu", Normal(0, 5))
        tau = sample("tau", HalfCauchy(scale=5))

        with plate("num_schools", self.num_schools):
            theta = sample("theta", Normal(mu, tau))
        sample(
            "y",
            Normal(theta, self.sigma),
            obs=obs["y"] if obs is not None else None,
        )


class EightSchoolsGuide(AbstractNumpyroGuide):
    """Eight schools guide using MLPs to parameterize simple distributions."""

    theta_base: AbstractDistribution
    mu_base: AbstractDistribution
    tau: AbstractDistribution

    def __init__(self, key, **kwargs):
        key, subkey = jr.split(key)
        self.mu_base = MLPParameterizedDistribution(
            subkey,
            Normal(),
            cond_dim=EightSchoolsModel.num_schools,
            **kwargs,
        )

        key, subkey = jr.split(key)
        self.tau = MLPParameterizedDistribution(
            subkey,
            HalfNormal(),
            cond_dim=EightSchoolsModel.num_schools,
            **kwargs,
        )

        key, subkey = jr.split(key)
        self.theta_base = MLPParameterizedDistribution(
            subkey,
            Normal(),
            cond_dim="scalar",
            **kwargs,
        )

    def __call__(
        self,
        obs: dict[str, Float[Array, " 8"]],
    ):
        sample("mu_base", self.mu_base, condition=obs.get("y"))
        sample("tau", self.tau, condition=obs.get("y"))

        with plate("num_schools", EightSchoolsModel.num_schools):
            sample("theta_base", self.theta_base, condition=obs.get("y"))


class EightSchoolsTask(AbstractTaskWithReference):
    guide: EightSchoolsGuide
    model = EightSchoolsModel()
    posterior_db_name = "eight_schools-eight_schools_noncentered"

    def __init__(self, key: PRNGKeyArray):
        self.guide = EightSchoolsGuide(key, width_size=20)

    def get_obs_and_latents(
        self,
        key: PRNGKeyArray | None = None,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Get the observations and a reference posterior from posteriordb.

        Key is ignored, but provided for consistency of API.
        """
        obs = {"y": jnp.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)}
        parameters = get_posterior_db_reference_posterior(self.posterior_db_name)
        return obs, parameters
