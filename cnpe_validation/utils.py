import warnings
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from flowjax.distributions import AbstractDistribution
from flowjax.wrappers import NonTrainable, unwrap
from jax import Array, vmap
from jax.flatten_util import ravel_pytree


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
        cond_dim: int | str,
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
        dist = self.constructor(self.mlp(condition))
        return dist.sample(key)

    def _log_prob(self, x, condition):
        dist = self.constructor(self.mlp(condition))
        return dist.log_prob(x)


def get_abspath_project_root():
    return Path(__file__).parent.parent
