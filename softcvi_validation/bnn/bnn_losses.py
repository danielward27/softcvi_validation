"""These losses are equivilent to those used in the main experiments.

We reimplement the losses for the additive BNN experiment to avoid the friction of
defining the model in a NumPyro compatible way, which although possible, is clunky.
"""

from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.lax import stop_gradient
from jaxtyping import Array, PRNGKeyArray, Scalar
from optax.losses import softmax_cross_entropy

from softcvi_validation.bnn.bnn import AdditiveBayesianMLP
from softcvi_validation.bnn.bnn_tasks import BNNRegressionProblem


@eqx.filter_jit
def contrastive_loss(
    params: AdditiveBayesianMLP,
    static: AdditiveBayesianMLP,
    key: PRNGKeyArray,
    x: Array,
    y: Array,
    *,
    alpha,
    prior: AdditiveBayesianMLP,
    k=2,
) -> Scalar:
    guide = eqx.combine(params, static)
    proposal = eqx.combine(stop_gradient(params), static)

    @eqx.filter_vmap
    def get_log_probs(key):
        samp = proposal.sample(key)
        proposal_lp = proposal.log_prob(samp)
        prior_lp = prior.log_prob(samp)
        guide_lp = guide.log_prob(samp)
        y_hat = eqx.filter_vmap(samp)(x)
        positive_lp = prior_lp + BNNRegressionProblem.log_likelihood(y_hat, y)
        negative_lp = proposal_lp * alpha
        return {
            "positive": positive_lp,
            "negative": negative_lp,
            "guide": guide_lp,
        }

    lps = get_log_probs(jr.split(key, k))
    labels = nn.softmax(lps["positive"] - lps["negative"])
    log_predictions = nn.log_softmax(lps["guide"] - lps["negative"])
    return softmax_cross_entropy(log_predictions, labels).mean()


@eqx.filter_jit
def elbo_loss(
    params: AdditiveBayesianMLP,
    static: AdditiveBayesianMLP,
    key: PRNGKeyArray,
    x: Array,
    y: Array,
    *,
    prior: AdditiveBayesianMLP,
    k=2,
) -> Scalar:
    def elbo_single(key):
        guide = eqx.combine(params, static)
        mlp = guide.sample(key)
        q_lp = guide.log_prob(mlp)
        prior_lp = prior.log_prob(mlp)
        y_hat = eqx.filter_vmap(mlp)(x)
        model_lp = prior_lp + BNNRegressionProblem.log_likelihood(y_hat, y)
        return -(model_lp - q_lp)

    return eqx.filter_vmap(elbo_single)(jr.split(key, k)).mean()


@eqx.filter_jit
def snisfkl_loss(
    params: AdditiveBayesianMLP,
    static: AdditiveBayesianMLP,
    key: PRNGKeyArray,
    x: Array,
    y: Array,
    *,
    prior: AdditiveBayesianMLP,
    k=2,
) -> Scalar:
    q = eqx.combine(params, static)
    proposal = eqx.combine(stop_gradient(params), static)

    @eqx.filter_vmap
    def get_log_probs(key):
        samp = proposal.sample(key)
        proposal_lp = proposal.log_prob(samp)
        prior_lp = prior.log_prob(samp)
        q_lp = q.log_prob(samp)
        y_hat = eqx.filter_vmap(samp)(x)
        joint_lp = prior_lp + BNNRegressionProblem.log_likelihood(y_hat, y)
        return {
            "joint": joint_lp,
            "proposal": proposal_lp,
            "guide": q_lp,
        }

    lps = get_log_probs(jr.split(key, k))
    log_weights = lps["joint"] - lps["proposal"]
    normalized_weights = nn.softmax(log_weights)
    return jnp.sum(normalized_weights * (lps["joint"] - lps["guide"]))


def get_losses(prior, k):
    shared_kwargs = {"k": k, "prior": prior}

    return {
        "SoftCVI(a=0.75)": partial(
            contrastive_loss,
            alpha=0.75,
            **shared_kwargs,
        ),
        "SoftCVI(a=1)": partial(
            contrastive_loss,
            alpha=1,
            **shared_kwargs,
        ),
        "SNIS-fKL": partial(
            snisfkl_loss,
            **shared_kwargs,
        ),
        "ELBO": partial(
            elbo_loss,
            **shared_kwargs,
        ),
    }
