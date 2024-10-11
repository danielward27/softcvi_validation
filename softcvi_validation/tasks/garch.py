"""A Generalized Autoregressive Conditional heteroscedasticit GARCH(1,1) task."""

from functools import partial
from typing import ClassVar

import jax.numpy as jnp
import numpyro.distributions as ndist
from flowjax.bijections import Invert, RationalQuadraticSpline
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
from numpyro.distributions import constraints
from numpyro.infer.reparam import TransformReparam
from pyrox.program import AbstractProgram, ReparameterizedProgram

from softcvi_validation.distributions import MLPParameterizedDistribution
from softcvi_validation.tasks.tasks import AbstractTaskWithFileReference


class GARCHModel(AbstractProgram):
    """The GARCH(1,1) model."""

    reparameterized: bool | None
    sigma1: ClassVar[float] = 0.5
    time: ClassVar[int] = 200

    def __init__(self):
        self.reparameterized = None

    def call_without_reparam(
        self,
        obs: dict[str, Float[Array, " 200"]] | None = None,
    ):
        mu = sample("mu", ndist.ImproperUniform(constraints.real, (), ()))
        alpha0 = sample("alpha0", ndist.ImproperUniform(constraints.positive, (), ()))

        alpha1_dist = Uniform(0, 1)
        alpha1 = sample("alpha1", alpha1_dist)

        beta1_dist = Uniform(0, jnp.clip(1 - alpha1, min=1e-8))  # Clip for stability.
        beta1 = sample("beta1", beta1_dist)

        def step_fn(carry, y):
            sigma, y_current = carry
            sigma = jnp.sqrt(alpha0 + alpha1 * (y_current - mu) ** 2 + beta1 * sigma**2)
            y = sample("y", ndist.Normal(mu, sigma), obs=y)
            return (sigma, y), (sigma, y)

        step_fn = partial(step_fn)

        y0 = obs["y"][0] if obs is not None else mu
        xs = None if obs is None else obs["y"]
        scan(step_fn, init=(jnp.array(self.sigma1), y0), xs=xs, length=self.time)


class GARCHGuide(AbstractProgram):
    """The guide used for the GARCH task.

    The guide uses a combination of simple distributions (for mu and alpha0), in
    addition to more flexible distributions (uniforms transformed with splines)
    for alpha1 and beta1.

    Args:
        key: Jax random key.
    """

    mu: Normal
    alpha0: LogNormal
    alpha1: Transformed
    beta1_base: MLPParameterizedDistribution

    def __init__(self, key: PRNGKeyArray):
        self.mu = Normal()
        self.alpha0 = LogNormal()

        self.alpha1 = Transformed(
            non_trainable(Uniform(0, 1)),
            Invert(RationalQuadraticSpline(knots=10, interval=(0, 1))),
        )
        self.beta1_base = MLPParameterizedDistribution(
            key,
            Transformed(
                non_trainable(Uniform(0, 1)),
                Invert(RationalQuadraticSpline(knots=10, interval=(0, 1))),
            ),
            cond_dim=2,  # alpha0, alpha1
            width_size=20,
            final_activation=lambda x: x / 5,  # At initialization prefer smaller vals
        )

    def __call__(self, obs: dict[str, Array] | None = None):
        sample("mu", self.mu)
        alpha0 = sample("alpha0", self.alpha0)
        alpha1 = sample("alpha1", self.alpha1)
        sample("beta1_base", self.beta1_base, condition=jnp.stack((alpha0, alpha1)))


class GARCHTask(AbstractTaskWithFileReference):
    """Generalized autoregressive conditional heteroskedasticity (GARCH) task.

    For more information, see Bollerslev, 1986,
    https://doi.org/10.1016/0304-4076(86)90063-1.
    """

    model: GARCHModel
    guide: GARCHGuide
    name = "garch"
    learning_rate = 5e-4

    def __init__(self, key: PRNGKeyArray):
        self.model = ReparameterizedProgram(
            GARCHModel(),
            config={"beta1": TransformReparam},
        )
        self.guide = GARCHGuide(key)
