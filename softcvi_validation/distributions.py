"""Extra flowjax distributions."""

from collections.abc import Callable
from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import Chain, Loc, Scale, SoftPlus, Tanh
from flowjax.distributions import (
    AbstractDistribution,
    AbstractTransformed,
    Logistic,
)
from flowjax.utils import arraylike_to_array, inv_softplus
from flowjax.wrappers import AbstractUnwrappable, NonTrainable, Parameterize, unwrap
from jax.flatten_util import ravel_pytree
from jax.nn import softplus
from jax.scipy.special import logsumexp
from jax.scipy.stats import truncnorm
from jaxtyping import Array, ArrayLike, PRNGKeyArray


class TruncNormal(AbstractDistribution):
    """Truncated normal distribution.

    The location and scale are trainable but the interval (the support) is not (assuming
    float/ints are filtered out).

    Args:
        lower: Lower bound of support.
        upper: Upper bound of support.
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    loc: Array
    scale: Array | AbstractUnwrappable[Array]
    lower: int | float
    upper: int | float
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(
        self,
        lower: float | int,
        upper: float | int,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
    ):
        self.loc, scale = jnp.broadcast_arrays(
            arraylike_to_array(loc, dtype=float),
            arraylike_to_array(scale, dtype=float),
        )
        self.scale = Parameterize(softplus, inv_softplus(scale))
        self.lower, self.upper = lower, upper
        self.shape = self.loc.shape

    def _log_prob(self, x, condition=None):
        a = (self.lower - self.loc) / self.scale
        b = (self.upper - self.loc) / self.scale
        return truncnorm.logpdf(x, a, b, self.loc, self.scale)

    def _sample(self, key, condition=None):
        standard_tnorm = jr.truncated_normal(
            key,
            lower=self.lower / self.scale - self.loc / self.scale,
            upper=self.upper / self.scale - self.loc / self.scale,
            shape=self.shape,
        )
        return standard_tnorm * self.scale + self.loc


class UniformWithLogisticBase(AbstractTransformed):
    """A uniform distribution parameterized as a transformed logistic distribution.

    We use this in models so we can use TransformReparam, such that the variational
    distribution can be learned on the unbounded (standard logistically distributed)
    space.
    """

    bijection: Chain
    base_dist: Logistic

    def __init__(self, minval: ArrayLike = 0, maxval: ArrayLike = 1):
        minval, maxval = jnp.broadcast_arrays(minval, maxval)
        shape = jnp.shape(minval)
        # Tanh maps logistic(scale=0.5) to uniform on [-1, 1]
        self.base_dist = Logistic(scale=jnp.full(shape, 0.5))
        scale_vals = (maxval - minval) / 2
        scale = Scale(jnp.ones_like(scale_vals))
        # Replace scale to avoid reparameterization
        scale = eqx.tree_at(lambda scale: scale.scale, scale, scale_vals)
        self.bijection = Chain([Tanh(shape), scale, Loc(minval + scale_vals)])


class Folded(AbstractDistribution):
    """Create a folded distribution.

    Folded distributions "fold" the probability mass from below the origin,
    to above it. This correspond to the absolute value transform
    for the distribution. Note if the distribution is symmetric and centered at zero,
    this corresponds to a half distribution.
    """

    dist: AbstractDistribution
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def __init__(self, dist: AbstractDistribution):
        if dist.shape != ():
            raise ValueError("Non-scalar distributions not yet supported with Folded.")

        self.dist = dist
        self.shape = dist.shape
        self.cond_shape = dist.cond_shape

    def _sample(self, key: PRNGKeyArray, condition=None):
        return jnp.abs(self.dist._sample(key, condition))

    def _log_prob(self, x, condition=None):
        abs_x = jnp.abs(x)
        above_below = jnp.stack((abs_x, -abs_x))
        unfolded_probs = self.dist.log_prob(above_below, condition=condition)
        return jnp.where(x < 0, -jnp.inf, logsumexp(unfolded_probs))


class MLPParameterizedDistribution(AbstractDistribution):
    """Parameterize a distribution using a neural network.

    Args:
        key: Jax random key.
        distribution: Distribution to parameterize with neural network.
        cond_dim: Int for integer dimension or "scalar".
        width_size: Neural network width.
        depth: Neural network depth. Defaults to 1.
        **kwargs: Key word arguments passed to ``equinox.nn.MLP``.

    """

    constructor: Callable
    mlp: eqx.nn.MLP
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]

    def __init__(
        self,
        key: Array,
        distribution: AbstractDistribution,
        *,
        cond_dim: int | Literal["scalar"],
        width_size,
        depth: int = 1,
        **kwargs,
    ):

        if unwrap(distribution).cond_shape is not None:
            raise ValueError("Expected unconditional distribution.")

        def get_constructor_and_num_params(distribution):
            params, static = eqx.partition(
                distribution,
                eqx.is_inexact_array,
                is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
            )
            init, unravel = ravel_pytree(params)

            def constructor(ravelled_params: Array):
                params = unravel(ravelled_params + init)
                return eqx.combine(params, static)

            return constructor, len(init)

        self.constructor, num_params = get_constructor_and_num_params(distribution)
        self.mlp = eqx.nn.MLP(
            cond_dim,
            out_size=num_params,
            width_size=width_size,
            depth=depth,
            key=key,
            **kwargs,
        )
        self.shape = unwrap(distribution).shape
        self.cond_shape = () if cond_dim == "scalar" else (cond_dim,)

    def _sample(self, key, condition):
        return self.to_dist(condition).sample(key)

    def _log_prob(self, x, condition):
        return self.to_dist(condition).log_prob(x)

    def to_dist(self, condition):
        return self.constructor(self.mlp(condition))


class _PositiveImproperUniformBase(AbstractDistribution):
    shape: tuple[int, ...]
    cond_shape: ClassVar[tuple[int, ...]] = None

    def _log_prob(self, x, condition=None):
        return -softplus(-x).sum()

    def _sample(self, key, condition=None):
        raise NotImplementedError()


class PositiveImproperUniform(AbstractTransformed):
    """Improper uniform distribution over the positive space."""

    base_dist: _PositiveImproperUniformBase
    bijection: SoftPlus

    def __init__(self, shape: tuple[int, ...] = ()):
        self.base_dist = _PositiveImproperUniformBase(shape)
        self.bijection = SoftPlus(shape)
