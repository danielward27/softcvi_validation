import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro
from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel
from flowjax.distributions import (
    AbstractDistribution,
    MultivariateNormal,
    Normal,
)
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow
from jaxtyping import Array, Float, PRNGKeyArray

from cnpe_validation.distributions import UniformWithLogisticBase
from cnpe_validation.tasks.tasks import AbstractTaskWithFileReference


class SLCPModel(AbstractNumpyroModel):
    reparameterized: bool | None = None
    observed_names = {"x"}
    reparam_names = {"theta"}

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


class SLCPGuide(AbstractNumpyroGuide):
    theta_base: AbstractDistribution
    embedding_net: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        key1, key2 = jr.split(key)
        # Obs are exchangeable so we embed with exchangeable nn.
        self.embedding_net = eqx.nn.MLP(
            key=key1,
            in_size=2,
            out_size=8,
            width_size=50,
            depth=1,
        )
        self.theta_base = masked_autoregressive_flow(
            key=key2,
            base_dist=Normal(jnp.zeros((5,))),
            cond_dim=8,
            nn_width=100,
        )

    def __call__(self, obs: dict[str, Float[Array, "4 2"]]):
        obs = jnp.tanh(obs["x"] / 20) * 5
        embedding = eqx.filter_vmap(self.embedding_net)(obs).mean(axis=0)
        sample("theta_base", self.theta_base, condition=embedding)


class SLCPTask(AbstractTaskWithFileReference):
    guide: SLCPGuide
    model = SLCPModel()
    name = "slcp"

    def __init__(self, key: PRNGKeyArray):
        self.guide = SLCPGuide(key)
