"""The eight schools task."""

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
from jax import Array
from jaxtyping import Float, ScalarLike
from numpyro.infer.reparam import TransformReparam
from paramax.wrappers import NonTrainable
from pyrox.program import AbstractProgram, ReparameterizedProgram

from softcvi_validation.distributions import Folded
from softcvi_validation.tasks.tasks import AbstractTaskWithFileReference


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


class EightSchoolsModel(AbstractProgram):
    """Eight schools model."""

    num_schools: ClassVar[int] = 8
    sigma: ClassVar[Array] = jnp.array([15, 10, 16, 11, 9, 11, 10, 18])

    def __call__(
        self,
        obs: Float[Array, " 8"] | None = None,
    ):
        mu = sample("mu", Normal(0, 5))
        tau = sample("tau", get_folded_distribution(Cauchy, loc=0, scale=5))
        theta = sample("theta", Normal(jnp.full((8,), mu), tau))
        sample("y", Normal(theta, self.sigma), obs=obs)


class EightSchoolsGuide(AbstractProgram):
    """Eight schools guide."""

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
    """The eight schools task.

    A classic hierarchical Bayesian inference problem, where the aim is to infer the
    treatment effects of a coaching program applied to eight schools, which are assumed
    to be exchangeable.

    Ref:
        Donald B Rubin. “Estimation in parallel randomized experiments”. In: Journal of
            Educational Statistics 6.4 (1981), pp. 377–401.
        Andrew Gelman et al. Bayesian data analysis. Chapman and Hall/CRC, 1995.

    Args:
        key: Ignored, but provided for consistency of API.
    """

    model: ReparameterizedProgram
    guide: EightSchoolsGuide
    name = "eight_schools"
    learning_rate = 1e-3
    observed_name = "y"
    latent_names = set({"mu", "tau", "theta"})

    def __init__(self, key):
        self.model = ReparameterizedProgram(
            EightSchoolsModel(),
            config={param: TransformReparam() for param in ("mu", "theta", "tau")},
        )
        self.guide = EightSchoolsGuide()
