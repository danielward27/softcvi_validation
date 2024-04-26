from collections.abc import Sequence

import jax.random as jr
from jax import vmap
from numpyro import handlers
from numpyro.infer import Predictive, log_likelihood


def sample_posterior(key, guide, obs, n: int, model=None):
    """Sample the posterior. If a model is provided, we also sample the predictive."""
    key1, key2 = jr.split(key)
    posterior = Predictive(guide, num_samples=n)
    posterior_samps = posterior(key1, obs=obs)
    if model is not None:
        predictive = Predictive(model, posterior_samples=posterior_samps)
        posterior_samps = posterior_samps | predictive(key2)
    return posterior_samps


def log_likelihood_obs(key, guide, model, obs, n: int):
    """Sample the posterior and compute the average likelihood of obs."""
    posterior_samps = sample_posterior(key, guide, obs, n)
    log_lik = log_likelihood(
        model=model,
        posterior_samples=posterior_samps,
        obs=obs,
    ).values()
    assert len(log_lik) == 1
    return list(log_lik.values())[0].mean()


def log_likelihood_held_out(
    key,
    model,
    guide,
    true_latents,
    n: int,
    observed_names: Sequence[str],
):
    """Note that the true latents provided should be the subset that are global."""

    def held_out_log_likeliood_single(key):
        # Sample 1 dataset, and 1 posterior sample and compute log likelihood
        observed_key, posterior_key = jr.split(key)
        held_out_obs = handlers.trace(
            handlers.substitute(handlers.seed(model, observed_key)),
            true_latents,
        )
        held_out_obs = {k: held_out_obs[k] for k in observed_names}

        return log_likelihood_obs(posterior_key, guide, model, held_out_obs, 1)

    return vmap(held_out_log_likeliood_single)(jr.split(key, n)).mean()
