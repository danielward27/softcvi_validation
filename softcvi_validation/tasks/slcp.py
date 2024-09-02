"""Simple likelihood complex posterior (SLCP) task."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import (
    AbstractDistribution,
    MultivariateNormal,
    Uniform,
)
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow
from flowjax.wrappers import NonTrainable, non_trainable
from jaxtyping import Array, Float, PRNGKeyArray
from softcvi.models import AbstractGuide, AbstractReparameterizedModel

from softcvi_validation.tasks.tasks import AbstractTaskWithFileReference


class SLCPModel(AbstractReparameterizedModel):
    """The model for the SLCP task."""

    reparameterized: bool | None = None
    observed_names = frozenset({"x"})
    reparam_names = frozenset(set())
    interval: int = 3

    def call_without_reparam(self, obs: dict[str, Float[Array, "4 2"]] | None = None):
        obs = obs["x"] if obs is not None else None
        theta = sample("theta", Uniform(jnp.full((5,), -self.interval), self.interval))
        mu = theta[:2]
        scale1 = theta[2] ** 2
        scale2 = theta[3] ** 2
        p = jnp.tanh(theta[4])
        cov = jnp.array(
            [[scale1**2, p * scale1 * scale2], [p * scale1 * scale2, scale2**2]],
        )
        with numpyro.plate("n_obs", 4):
            sample("x", MultivariateNormal(mu, cov), obs=obs)


class SLCPGuide(AbstractGuide):
    """The guide used for the SLCP task.

    The guide is a masked autoregressive flow, with a rational quadratic spline
    transformer.

    Args:
        key: Jax random seed.
    """

    theta: AbstractDistribution

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        base_dist = Uniform(jnp.full((5,), -SLCPModel.interval), SLCPModel.interval)

        flow = masked_autoregressive_flow(
            key=key,
            base_dist=non_trainable(base_dist),  # Don't optimize uniform!
            nn_width=30,
            flow_layers=4,
            transformer=RationalQuadraticSpline(knots=10, interval=SLCPModel.interval),
        )
        flow = jax.tree.map(  # Use smaller vals on init (closer to identity)
            lambda leaf: leaf / 5 if eqx.is_inexact_array(leaf) else leaf,
            flow,
            is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
        )
        self.theta = flow

    def __call__(self, obs: dict[str, Array] | None = None):
        sample("theta", self.theta)


class SLCPTask(AbstractTaskWithFileReference):
    """Simple Likelihood Complex Posterior (SLCP) task.

    A multivariate Gaussian is parameterized in a manner that induces a complex
    multimodal posterior. For more information, see Papamakarios et al., 2019,
    https://arxiv.org/abs/1805.07226).

    Args:
        key: The jax random seed used to initialize the guide.
    """

    guide: SLCPGuide
    model: SLCPModel
    name = "slcp"
    learning_rate = 1e-4

    def __init__(self, key: PRNGKeyArray):
        self.model = SLCPModel()
        self.guide = SLCPGuide(key)
