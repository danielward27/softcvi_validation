"""Posterior performance metrics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray
from softcvi.models import AbstractGuide, AbstractReparameterizedModel


def coverage_probabilities(
    key: PRNGKeyArray,
    *,
    model: AbstractReparameterizedModel,
    guide: AbstractGuide,
    obs: dict,
    reference_samples: dict[str, Array],
    n_samps: int = 5000,
    nominal_percentiles: Float[Array, " n"] | None = None,
):
    """Compute the coverage probabilties given a guide and a set of reference samples.

    This uses the density quantile approach for inferring the higestest posterior
    density region:
        - Hyndman (1996), Computing and Graphing Highest Density Regions, JSTOR.

    Args:
        key: Jax random key.
        model: The model (required for infering reparmaeterizations).
        guide: The guide.
        obs: The set of observations.
        reference_samples: The set of reference samples.
        n_samps: How many guide samples to use for inferring the highest density region.
            Defaults to 5000.
        nominal_percentiles: The nominal percentiles of the credible region to consider.
            Defaults to ``jnp.linspace(0, 100, 100)``.
    """
    if nominal_percentiles is None:
        nominal_percentiles = jnp.linspace(0, 100, 100)

    @eqx.filter_jit
    @_map_wrapper
    def sample_guide_original_space(key):
        guide_samp = guide.sample(key, obs=obs)
        return model.latents_to_original_space(guide_samp, obs=obs)

    @eqx.filter_jit
    @_map_wrapper
    def log_prob_original_space(latents):
        return guide.log_prob_original_space(latents, model=model, obs=obs)

    key, subkey = jr.split(key)
    guide_samples = sample_guide_original_space(jr.split(subkey, n_samps))
    guide_log_probs = jnp.sort(log_prob_original_space(guide_samples))
    ref_log_probs = log_prob_original_space(reference_samples)

    # Compute percentile of reference log probs, within guide_log_probs
    percentile = (
        100 * jnp.searchsorted(guide_log_probs, ref_log_probs) / len(guide_log_probs)
    )

    # Smallest nominal percentile required such that reference sample in credible region
    smallest_required_nominal = 100 - percentile

    def _coverage_frequency(percentile):
        return (percentile >= smallest_required_nominal).mean()

    return eqx.filter_vmap(_coverage_frequency)(nominal_percentiles)


def negative_posterior_mean_l2(
    key: PRNGKeyArray,
    *,
    model: AbstractReparameterizedModel,
    guide: AbstractGuide,
    obs: dict,
    reference_samples: dict[str, Array],
    n_samps: int = 5000,
):
    """Calculates the posterior mean accuracy.

    This is the negative l2-norm of the difference between the means of the approximate
    and reference samples, after scaling by the scale of the reference samples.
    """

    @eqx.filter_vmap
    def _to_matrix(samples: dict):
        return ravel_pytree(samples)[0]

    @eqx.filter_jit
    @_map_wrapper
    def sample_guide_original_space(key):
        guide_samp = guide.sample(key, obs=obs)
        return model.latents_to_original_space(guide_samp, obs=obs)

    key, subkey = jr.split(key)
    guide_samples = sample_guide_original_space(jr.split(subkey, n_samps))
    guide_samples = _to_matrix(guide_samples)
    reference_samples = _to_matrix(reference_samples)

    # Get an estimate of parameter scales
    scales = jnp.std(reference_samples, axis=0)
    guide_samples = guide_samples / scales
    reference_samples = reference_samples / scales
    return -jnp.linalg.norm(guide_samples.mean(axis=0) - reference_samples.mean(axis=0))


def mean_log_prob_reference(
    model: AbstractReparameterizedModel,
    guide: AbstractGuide,
    obs: dict[str, Array],
    reference_samples: dict[str, Array],
):
    """Calculate the mean log probability of the reference samples, in the guide.

    Args:
        model: The model.
        guide: The guide.
        obs: Dictionary of observations.
        reference_samples: The reference samples, with a leading batch dimension.
    """

    @_map_wrapper
    def _log_prob(samps):
        return guide.log_prob_original_space(samps, model=model, obs=obs)

    return _log_prob(reference_samples).mean()


def _map_wrapper(f):  # We use map instead of vmap just incase memory would be an issue
    return lambda xs: jax.lax.map(f, xs)
