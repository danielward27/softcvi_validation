# %%
import jax.numpy as jnp
import numpyro
from cpe.models import AbstractGuide, AbstractModel
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import (
    AbstractDistribution,
    MultivariateNormal,
    Normal,
)
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow
from jaxtyping import Array, Float, PRNGKeyArray

from cpe_validation.distributions import UniformWithLogisticBase
from cpe_validation.tasks.tasks import AbstractTaskWithFileReference


class SLCPModel(AbstractModel):
    reparameterized: bool | None = None
    observed_names = frozenset({"x"})
    reparam_names = frozenset({"theta"})

    def call_without_reparam(self, obs: dict[str, Float[Array, "4 2"]] | None = None):
        obs = obs["x"] if obs is not None else None
        theta = sample("theta", UniformWithLogisticBase(jnp.full((5,), -3), 3))
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
    # Key accepted for consistency of API
    theta_base: AbstractDistribution

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        self.theta_base = masked_autoregressive_flow(
            key=key,
            base_dist=Normal(jnp.zeros((5,)), 0.75),
            nn_width=20,
            transformer=RationalQuadraticSpline(knots=10, interval=4),
        )

    def __call__(self):
        sample("theta_base", self.theta_base)


class SLCPTask(AbstractTaskWithFileReference):
    guide: SLCPGuide
    model: SLCPModel
    name = "slcp"

    def __init__(self, key: PRNGKeyArray):
        self.model = SLCPModel()
        self.guide = SLCPGuide(key)
