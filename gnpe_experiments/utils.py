import json
import warnings
import zipfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import requests
from flowjax.distributions import AbstractDistribution
from jax import Array
from jax.flatten_util import ravel_pytree


def drop_nan_and_warn(*arrays, axis: int):

    shapes_except_last = [a.shape[:-1] for a in arrays]
    assert all(s == shapes_except_last[0] for s in shapes_except_last)

    is_finite = [jnp.isfinite(a).all(axis=-1).squeeze() for a in arrays]
    is_finite = jnp.stack(is_finite, axis=-1).all(axis=-1)

    if not jnp.all(is_finite):
        warnings.warn(
            f"Dropping {(~is_finite).sum()} nan or inf values.",
            stacklevel=1,
        )
        arrays = [jnp.compress(is_finite, a, axis) for a in arrays]
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

        if distribution.cond_shape is not None:
            raise ValueError("Expected unconditional distribution.")

        def get_constructor_and_num_params(distribution):
            params, static = eqx.partition(distribution, eqx.is_inexact_array)
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
        self.shape = distribution.shape
        self.cond_shape = () if cond_dim == "scalar" else (cond_dim,)

    def _sample(self, key, condition):
        dist = self.constructor(self.mlp(condition))
        return dist.sample(key)

    def _log_prob(self, x, condition):
        dist = self.constructor(self.mlp(condition))
        return dist.log_prob(x)


def get_abspath_project_root():
    return Path(__file__).parent.parent


def get_posterior_db_reference_posterior(name) -> dict:
    """Get the reference posterior draws from posteriordb.

    https://github.com/stan-dev/posteriordb

    Args:
        name: The name of the zip file containing the draws, exluding the extension.
    """
    # Targetting tagged release 0.5.0 for better reproducibility
    url = f"https://github.com/stan-dev/posteriordb/raw/0.5.0/posterior_database/reference_posteriors/draws/draws/{name}.json.zip"

    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()
    zip_content = BytesIO(response.content)

    # Extract the zip file
    with zipfile.ZipFile(zip_content, "r") as zip_ref:
        assert len(zip_ref.infolist()) == 1
        zip_info = zip_ref.infolist()[0]

        # Extract the JSON file from the zip
        with zip_ref.open(zip_info) as json_file:
            # Read the JSON data
            draws = json.load(json_file)

    # Concatenate the chains
    draws = {
        k: jnp.concatenate([jnp.asarray(chain[k]) for chain in draws])
        for k in draws[0].keys()
    }

    # Names may be of form param[1] param[2]. We want to stack these into an array
    stacked_draws = {}
    for k, v in draws.items():
        key_root = k.split("[")[0]

        if key_root in stacked_draws:
            stacked_draws[key_root] = jnp.stack((stacked_draws[key_root], v), axis=-1)
        else:
            stacked_draws[key_root] = v

    return stacked_draws
