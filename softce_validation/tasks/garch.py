# %%
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpyro.distributions as ndist
from flowjax.bijections import Chain, Loc, RationalQuadraticSpline
from flowjax.distributions import (
    LogNormal,
    Normal,
    Transformed,
    Uniform,
)
from flowjax.experimental.numpyro import sample
from flowjax.wrappers import non_trainable
from jaxtyping import Array, Float, PRNGKeyArray
from numpyro.contrib.control_flow import scan
from softce.models import AbstractGuide, AbstractModel

from softce_validation.distributions import MLPParameterizedDistribution
from softce_validation.tasks.tasks import AbstractTaskWithFileReference


class GARCHModel(AbstractModel):
    reparameterized: bool | None
    observed_names = {"y"}
    reparam_names = {"beta1"}
    sigma1: ClassVar[float] = 0.5
    time: ClassVar[int] = 200

    def __init__(self):
        self.reparameterized = None

    def call_without_reparam(
        self,
        obs: dict[str, Float[Array, " 200"]] | None = None,
    ):
        mu = sample(
            "mu",
            ndist.ImproperUniform(ndist.constraints.real, (), ()),
        )

        alpha0 = sample(
            "alpha0",
            ndist.ImproperUniform(ndist.constraints.positive, (), ()),
        )
        alpha1 = sample(
            "alpha1",
            Uniform(0, 1),
        )
        beta1 = sample(
            "beta1",
            Uniform(0, 1 - alpha1),
        )

        def step_fn(carry, y):
            sigma, y_current = carry
            sigma = jnp.sqrt(alpha0 + alpha1 * (y_current - mu) ** 2 + beta1 * sigma**2)
            y = sample("y", ndist.Normal(mu, sigma), obs=y)
            return (sigma, y), (sigma, y)

        y0 = obs["y"][0] if obs is not None else mu
        scan(
            step_fn,
            init=(jnp.array(self.sigma1), y0),
            xs=None if obs is None else obs["y"],
            length=self.time,
        )


class GARCHGuide(AbstractGuide):
    mu: Normal
    alpha0: LogNormal
    alpha1: Transformed
    beta1_base: MLPParameterizedDistribution

    def __init__(self, key: PRNGKeyArray):
        self.mu = Normal()
        self.alpha0 = LogNormal()

        def zero_one_spline_transformed_uniform():
            return Transformed(
                non_trainable(Uniform(-0.5, 0.5)),
                Chain(
                    [
                        RationalQuadraticSpline(knots=10, interval=0.5),
                        non_trainable(Loc(0.5)),
                    ],
                ),
            )

        self.alpha1 = zero_one_spline_transformed_uniform()
        self.beta1_base = MLPParameterizedDistribution(
            key,
            zero_one_spline_transformed_uniform(),
            cond_dim=2,  # alpha0, alpha1
            width_size=20,
        )

    def __call__(self):
        sample("mu", self.mu)
        alpha0 = sample("alpha0", self.alpha0)
        alpha1 = sample("alpha1", self.alpha1)
        sample("beta1_base", self.beta1_base, condition=jnp.stack((alpha0, alpha1)))


class GARCHTask(AbstractTaskWithFileReference):
    model: GARCHModel
    guide: GARCHGuide
    name = "garch"

    def __init__(self, key: PRNGKeyArray):
        self.model = GARCHModel()
        self.guide = GARCHGuide(key)
