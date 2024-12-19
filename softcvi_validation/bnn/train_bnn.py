"""Training loop for BNN."""

from collections.abc import Callable

import equinox as eqx
import jax.random as jr
import optax
import paramax
from flowjax.train.train_utils import step
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from softcvi_validation.bnn import AdditiveBayesianMLP
from softcvi_validation.bnn.bnn_metrics import test_set_log_likelihood


def train_bnn(
    key: PRNGKeyArray,
    bnn: AdditiveBayesianMLP,
    *,
    loss_fn: Callable[[PyTree, PyTree, PRNGKeyArray, Array, Array], Scalar],
    data: dict,
    steps: int,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    show_progress: bool = True,
    check_val_every: int = 100,
):
    """Train a pytree, using a loss with params, static and key as arguments.

    This can be used e.g. to fit a distribution using a variational objective, such as
    the evidence lower bound.

    Args:
        key: Jax random key.
        bnn: AdditiveBayesianMLP, from which trainable parameters are found using
            ``equinox.is_inexact_array``.
        loss_fn: The loss function to optimize, taking the partitioned model,
            random key, covariates and targets.
        data: The dictionary of data from the regression task.
        steps: The number of optimization steps.
        learning_rate: The adam learning rate. Ignored if optimizer is provided.
        optimizer: Optax optimizer. Defaults to None.
        show_progress: Whether to show progress bar. Defaults to True.
        check_val_every: How frequently to evaluate the validation loss. Note we
            do not do this every iteration as it is costly.

    Returns:
        A tuple containing the trained pytree and the validation log likelihoods.
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(
        bnn,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    opt_state = optimizer.init(params)
    keys = tqdm(jr.split(key, steps), disable=not show_progress)

    val_log_likelihoods = []

    for i, key in enumerate(keys):
        step_key, val_key = jr.split(key)

        # Train step
        params, opt_state, _ = step(
            params,
            static,
            step_key,
            data["train_x"],
            data["train_y"],
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
        )

        if i % check_val_every == 0:
            # Perform validation step every n iterations
            val_log_lik = eqx.filter_jit(test_set_log_likelihood)(
                key=val_key,
                bayesian_mlp=eqx.combine(params, static),
                x=data["val_x"],
                y=data["val_y"],
                n=100,
            ).item()
            val_log_likelihoods.append(val_log_lik)
            keys.set_postfix({"val log-likelihood": val_log_lik})

            if max(val_log_likelihoods) == val_log_lik:
                best_params = params

    return eqx.combine(best_params, static), val_log_likelihoods
