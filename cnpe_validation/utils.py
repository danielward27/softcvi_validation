import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import Chain, Loc, Scale, Tanh
from flowjax.distributions import (
    AbstractDistribution,
    AbstractTransformed,
    Logistic,
)
from flowjax.wrappers import NonTrainable, unwrap
from jax import Array, vmap
from jax.flatten_util import ravel_pytree
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree


def drop_nan_and_warn(*arrays):
    assert all(arrays[0].shape[0] == arr.shape[0] for arr in arrays)
    is_finite_fn = vmap(lambda arr: jnp.all(jnp.isfinite(arr)))
    is_finite = jnp.stack([is_finite_fn(a) for a in arrays], axis=-1).all(axis=-1)

    if not jnp.all(is_finite):
        warnings.warn(
            f"Dropping {(~is_finite).sum()} nan or inf values.",
            stacklevel=1,
        )
        arrays = [jnp.compress(is_finite, a, 0) for a in arrays]
    return arrays


def change_initialization(
    pytree: PyTree,
    key: PRNGKeyArray,
    initialization_fn: Callable,
    type_filter: type = PyTree,
):
    """Reinitialize a pytree/model.

    Type filter will only reinitilize particular subtypes within the pytree (by defult
    it is PyTree i.e. the entire) model.

    Args:
        pytree: The pytree to reinitialize.
        key: Jax PRNGKey.
        initialization_fn: Function taking key and shape (see jax.nn.initializers).
        type_filter: If provided, only submodules matching the type will be
            reinitialized.
    """
    to_reparam = eqx.filter(
        pytree,
        filter_spec=lambda leaf: isinstance(leaf, type_filter),
        is_leaf=lambda leaf: isinstance(leaf, type_filter),
    )
    params, static = eqx.partition(to_reparam, eqx.is_inexact_array)
    treedef = jax.tree_util.tree_structure(params)
    keys = jax.random.split(key, treedef.num_leaves)
    key_tree = jax.tree_unflatten(treedef, keys)
    params = jax.tree_util.tree_map(
        lambda leaf, key: initialization_fn(key, leaf.shape),
        params,
        key_tree,
    )
    return eqx.combine(eqx.combine(params, static), pytree)


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
        shape = minval.shape

        # Tanh maps logistic(scale=0.5) to uniform on [-1, 1]
        self.base_dist = Logistic(scale=jnp.full(shape, 0.5))
        scale = (maxval - minval) / 2
        self.bijection = Chain([Tanh(shape), Scale(scale), Loc(minval + scale)])


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


def get_abspath_project_root():
    return Path(__file__).parent.parent
