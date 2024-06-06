from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from flowjax.bijections import Scale
from flowjax.distributions import (
    AbstractDistribution,
    AbstractLocScaleDistribution,
    Cauchy,
    Normal,
    StudentT,
    Transformed,
)
from flowjax.experimental.numpyro import sample
from flowjax.utils import arraylike_to_array
from flowjax.wrappers import NonTrainable
from jax import Array
from jaxtyping import Float, ScalarLike
from softce.models import AbstractGuide, AbstractModel
from softce_validation.distributions import Folded
from softce_validation.tasks.tasks import AbstractTaskWithFileReference


def get_folded_distribution(
    dist_type: type[AbstractLocScaleDistribution],
    loc: ScalarLike = 0,
    scale: ScalarLike = 1,
    **kwargs,
):
    """Get a folded loc scale distribution.

    This is parameterized with the location transformation prior to folding, and the
    scale transformation after folding. Delaying the scale transformation can be useful
    for reparameterization. Note if loc is 0, this corresponds to a half
    distribution.
    """
    loc, scale = jnp.broadcast_arrays(
        *[arraylike_to_array(arr, dtype=float) for arr in [loc, scale]],
    )
    dist = dist_type(loc=loc, **kwargs)
    dist = eqx.tree_at(lambda n: n.bijection.scale, dist, replace_fn=NonTrainable)
    return Transformed(Folded(dist), Scale(scale))


class EightSchoolsModel(AbstractModel):
    """Eight schools model."""

    reparameterized: bool | None = None
    observed_names = {"y"}
    reparam_names = {"mu", "theta", "tau"}
    num_schools: ClassVar[int] = 8
    sigma: ClassVar[Array] = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])

    def call_without_reparam(
        self,
        obs: dict[str, Float[Array, " 8"]] | None = None,
    ):
        obs = obs["y"] if obs is not None else None
        mu = sample("mu", Normal(0, 5))
        tau = sample("tau", get_folded_distribution(Cauchy, loc=0, scale=5))
        theta = sample("theta", Normal(jnp.full((8,), mu), tau))
        sample("y", Normal(theta, self.sigma), obs=obs)


class EightSchoolsGuide(AbstractGuide):
    """Eight schools guide using MLPs to parameterize simple distributions."""

    theta_base: AbstractDistribution
    mu_base: AbstractDistribution
    tau_base: AbstractDistribution

    def __init__(self):
        self.mu_base = Normal()
        self.tau_base = get_folded_distribution(StudentT, df=5)
        self.theta_base = StudentT(df=jnp.full((8,), 5))

    def __call__(self):
        sample("mu_base", self.mu_base)
        sample("tau_base", self.tau_base)
        sample("theta_base", self.theta_base)


class EightSchoolsTask(AbstractTaskWithFileReference):
    model: EightSchoolsModel
    guide: EightSchoolsGuide
    name = "eight_schools"

    def __init__(self, key):
        # accepts key for consistency in API
        self.model = EightSchoolsModel()
        self.guide = EightSchoolsGuide()
