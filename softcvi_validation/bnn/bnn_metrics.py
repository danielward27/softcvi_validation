from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
from jaxtyping import Array, PRNGKeyArray

from softcvi_validation.bnn import AdditiveBayesianMLP
from softcvi_validation.bnn.bnn_tasks import BNNRegressionProblem


def compute_metrics(
    key: PRNGKeyArray,
    bayesian_mlp: AdditiveBayesianMLP,
    ground_truth_fn: Callable,
    data: dict,
):
    key, subkey = jr.split(key)
    tsl = test_set_log_likelihood(
        subkey,
        bayesian_mlp,
        data["test_x"],
        data["test_y"],
        n=1000,
    )
    key, subkey = jr.split(key)
    coverage = predictive_interval_coverage_and_width(
        key,
        bayesian_mlp,
        data["test_x"],
        ground_truth_fn,
        n=1000,
    )
    return {"test_set_log_likelihood": tsl} | coverage


def test_set_log_likelihood(
    key: PRNGKeyArray,
    bayesian_mlp: AdditiveBayesianMLP,
    x: Array,
    y: Array,
    n: int = 1000,
):

    @eqx.filter_vmap
    def eval_log_lik(key):
        key, subkey = jr.split(key)
        mlp = bayesian_mlp.sample(subkey)
        key, subkey = jr.split(key)
        idx = jr.randint(key, (), minval=0, maxval=len(y))
        y_hat = mlp(x[idx])
        return BNNRegressionProblem.log_likelihood(y_hat, y[idx])

    log_likelihoods = eval_log_lik(jr.split(key, n))
    return logsumexp(log_likelihoods - jnp.log(n))


def predictive_interval_coverage_and_width(
    key: PRNGKeyArray,
    bayesian_mlp: AdditiveBayesianMLP,
    x,
    ground_truth_fn: Callable,
    n: int = 1000,
):
    """The prediction interval coverage and width.

    Measures the average IQR of predictions across the test set, in addition to how
    frequently the true underlying function falls within the predicted IQRs.
    """
    covered = []
    width = []
    y_true = ground_truth_fn(x)
    for xi, y_true_i in zip(x, y_true, strict=True):
        key, subkey = jr.split(key)
        mlps = eqx.filter_vmap(bayesian_mlp.sample)(jr.split(subkey, n))
        y_hats = eqx.filter_vmap(lambda mlp: mlp(xi))(mlps)
        lower, upper = jnp.quantile(y_hats, jnp.array([0.25, 0.75]))
        covered.append((y_true_i < upper) & (y_true_i > lower))
        width.append(upper - lower)

    return {
        "mean_prediction_iqr": jnp.asarray(width).mean(),
        "mean_iqr_coverage": jnp.asarray(covered).mean(),
    }
