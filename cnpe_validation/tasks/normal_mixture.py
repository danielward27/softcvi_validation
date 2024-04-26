# %%
from typing import ClassVar

import jax.random as jr
from flowjax import bijections
from flowjax.distributions import (
    AbstractDistribution,
    AbstractLocScaleDistribution,
    AbstractTransformed,
)
from jax.scipy.stats import logistic
from jaxtyping import ArrayLike


# Into flowjax?
class _StandardLogistic(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _sample(self, key, condition=None):
        return jr.logistic(key, self.shape)

    def _log_prob(self, x, condition=None):
        return logistic.logpdf(x).sum()


class Logistic(AbstractLocScaleDistribution):
    base_dist: _StandardLogistic
    bijection: bijections.Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardLogistic(
            shape=jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = bijections.Affine(loc=loc, scale=scale)


class UniformWithLogisticBase(AbstractTransformed):
    """A uniform distribution parameterized as a transformed logistic distribution.

    We use this in models so we can use TransformReparam, such that the variational
    distribution can be learned on the unbounded (logistically distributed) space.
    """

    bijection: bijections.Tanh
    base_dist: Logistic

    def __init__(self, shape: tuple[int, ...] = ()):
        self.bijection = bijections.Tanh(shape)
        self.base_dist = Logistic(scale=0.5)
