import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel
from jaxtyping import Array, PRNGKeyArray


def coverage_probability(
    key: PRNGKeyArray,
    *,
    model: AbstractNumpyroModel,
    guide: AbstractNumpyroGuide,
    obs: dict,
    reference_samples: dict,
    n_guide_samps: int = 5000,
    nominal_percentiles: Array | None = None,
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
        n: How many guide samples to use for inferring the highest density region.
            Defaults to 5000.
        nominal_percentiles: The nominal percentiles of the credible region to consider.
            Defaults to ``jnp.linspace(0, 100, 100)``.
    """
    if nominal_percentiles is None:
        nominal_percentiles = jnp.linspace(0, 100, 100)

    def _map_wrapper(f):  # Use map instead of vmap to limit memory usage
        return lambda xs: jax.lax.map(f, xs)

    @eqx.filter_jit
    @_map_wrapper
    def sample_guide_original_space(key):
        guide_samp = guide.sample(key, obs=obs)
        return model.latents_to_original_space(guide_samp, obs=obs)

    @eqx.filter_jit
    @_map_wrapper
    def log_prob_original_space(latents):
        return guide.log_prob_original_space(latents, model, obs=obs)

    key, subkey = jr.split(key)
    guide_samples = sample_guide_original_space(jr.split(subkey, n_guide_samps))
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
