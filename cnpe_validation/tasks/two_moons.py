import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel
from flowjax.bijections import Chain, Loc, Scale
from flowjax.distributions import AbstractDistribution, Normal, Transformed
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow
from jaxtyping import Array, Float, PRNGKeyArray

from cnpe_validation.tasks.tasks import AbstractTaskWithoutReference
from cnpe_validation.utils import MLPParameterizedDistribution, UniformWithLogisticBase


class TwoMoonsModel(AbstractNumpyroModel):
    reparameterized: bool | None = None
    observed_names = {"x"}
    reparam_names = {"theta", "alpha"}

    def call_without_reparam(self, obs: dict[str, Float[Array, " 2"]] | None = None):
        theta = sample("theta", UniformWithLogisticBase(-jnp.ones(2), 1))
        alpha = sample("alpha", UniformWithLogisticBase(-jnp.pi / 2, jnp.pi / 2))

        scale = jnp.array([jnp.cos(alpha), jnp.sin(alpha)])
        loc = jnp.array(
            [
                0.25 - jnp.abs(jnp.sum(theta)) / 2**0.5,
                (-theta[0] + theta[1]) / 2**0.5,
            ],
        )

        # Scale by default reparameterized using softplus, so we construct in
        # this slightly awkward way to avoid this (we want to allow negative scales).
        scale_bijection = Scale(jnp.ones_like(scale))
        scale_bijection = eqx.tree_at(lambda scale: scale.scale, scale_bijection, scale)

        likelihood = Transformed(
            Normal(jnp.array([0.1, 0.1]), 0.01),
            Chain([scale_bijection, Loc(loc)]),
        )
        sample("x", likelihood, obs=obs["x"] if obs is not None else None)


class TwoMoonsGuide(AbstractNumpyroGuide):
    theta_base: AbstractDistribution
    alpha_base: AbstractDistribution

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        theta_key, alpha_key = jr.split(key)
        self.theta_base = masked_autoregressive_flow(
            key=theta_key,
            base_dist=Normal(jnp.zeros((2,))),
            cond_dim=2,
        )

        self.alpha_base = MLPParameterizedDistribution(
            alpha_key,
            Normal(),
            cond_dim=2,
            width_size=20,
        )  # Normal for now - but should consider posterior form

    def __call__(self, obs: dict[str, Float[Array, " 2"]]):
        sample("theta_base", self.theta_base, condition=obs["x"])
        sample("alpha_base", self.alpha_base, condition=obs["x"])


class TwoMoonsTask(AbstractTaskWithoutReference):  # or with...
    guide: TwoMoonsGuide
    model = TwoMoonsModel()
    name = "two_moons"

    def __init__(self, key: PRNGKeyArray):
        self.guide = TwoMoonsGuide(key)
