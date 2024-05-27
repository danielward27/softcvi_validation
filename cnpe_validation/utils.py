import warnings
from pathlib import Path

import jax.numpy as jnp
from jax import vmap


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


def get_abspath_project_root():
    return Path(__file__).parent.parent
