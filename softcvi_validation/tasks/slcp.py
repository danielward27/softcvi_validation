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
from flowjax.wrappers import non_trainable
from jaxtyping import Array, Float, PRNGKeyArray
from softcvi.models import AbstractGuide, AbstractModel
from softcvi_validation.tasks.tasks import AbstractTaskWithFileReference


class SLCPModel(AbstractModel):
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
    theta: AbstractDistribution

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        base_dist = Uniform(jnp.full((5,), -SLCPModel.interval), SLCPModel.interval)

        self.theta = masked_autoregressive_flow(
            key=key,
            base_dist=non_trainable(base_dist),  # Don't optimize uniform!
            nn_width=20,
            transformer=RationalQuadraticSpline(knots=10, interval=SLCPModel.interval),
        )

    def __call__(self):
        sample("theta", self.theta)


class SLCPTask(AbstractTaskWithFileReference):
    guide: SLCPGuide
    model: SLCPModel
    name = "slcp"

    def __init__(self, key: PRNGKeyArray):
        self.model = SLCPModel()
        self.guide = SLCPGuide(key)
