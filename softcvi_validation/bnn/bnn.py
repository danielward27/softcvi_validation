"""Bayesian neural network regression utilities.

We also reimplement the loss functions and utilities in a more general manner,
as here we do not use NumPyro. This isn't ideal and it's possible
to implement the below as a numpyro model, but it is inconvenient.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution
from jaxtyping import Array, PRNGKeyArray, PyTree


class AdditiveMLP(eqx.Module):
    """Additive MLP, made by vectorizing scalar MLPs."""

    mlps: eqx.nn.MLP

    def __init__(self, key, covariate_dim, **kwargs):

        @eqx.filter_vmap
        def get_mlps(key):
            return eqx.nn.MLP("scalar", "scalar", key=key, **kwargs)

        keys = jr.split(key, covariate_dim)
        self.mlps = get_mlps(keys)

    def __call__(self, x: Array):
        return jnp.sum(self.additive_components(x))

    def additive_components(self, x):
        return eqx.filter_vmap(lambda mlp, x: mlp(x))(self.mlps, x)


class AdditiveBayesianMLP(eqx.Module):
    amlp: AdditiveMLP

    def __init__(
        self,
        param_to_distribution: Callable,
        **kwargs,
    ):
        self.amlp = make_bayesian(AdditiveMLP(**kwargs), param_to_distribution)

    def sample(self, key: PRNGKeyArray) -> AdditiveMLP:
        keys = key_split_over_tree(key, self.amlp)

        def _map_fn(leaf, key):
            if isinstance(leaf, AbstractDistribution):
                return leaf.sample(key)
            return leaf

        return jax.tree_util.tree_map(
            _map_fn,
            self.amlp,
            keys,
            is_leaf=lambda leaf: isinstance(leaf, AbstractDistribution),
        )

    def log_prob(self, amlp: AdditiveMLP):

        def _map_fn(leaf1, leaf2):
            if isinstance(leaf1, AbstractDistribution):
                return leaf1.log_prob(leaf2)
            return None

        lps = jax.tree_util.tree_map(
            _map_fn,
            self.amlp,
            amlp,
            is_leaf=lambda leaf: isinstance(leaf, AbstractDistribution),
        )
        leaves = jax.tree_util.tree_leaves(lps)
        return sum(leaf for leaf in leaves)


def make_bayesian(
    tree: PyTree,
    param_to_distribution: Callable,
):
    """Replace all floating point arrays in a PyTree with a Gaussian."""

    def _map_fn(leaf):
        if eqx.is_inexact_array(leaf):
            return param_to_distribution(leaf)
        return leaf

    return jax.tree_util.tree_map(_map_fn, tree=tree)


def key_split_over_tree(key, target=None):
    treedef = jax.tree_util.tree_structure(
        target,
        is_leaf=lambda leaf: isinstance(leaf, AbstractDistribution),
    )
    keys = jax.random.split(key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)
