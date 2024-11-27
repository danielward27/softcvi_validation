"""Contains the script for running the inference algorithms for a Bayesian neural network (BNN) task.

From the project root directory, a run can be carried out using:

``python -m scripts.run_bnn --seed=0 --n-particles=8 --steps=10000 --loss-name="ELBO" --width-size=50 --learning-rate=1e-3``
"""

import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping
from flowjax.distributions import (
    Laplace,
    Normal,
)

with jaxtyping.install_import_hook(
    ["softcvi", "softcvi_validation", "pyrox"],
    "beartype.beartype",
):
    from softcvi_validation import bnn, utils
    from softcvi_validation.bnn import bnn_metrics, bnn_tasks
    from softcvi_validation.bnn.bnn_losses import get_losses
    from softcvi_validation.bnn.train_bnn import train_bnn
    from softcvi_validation.utils import get_abspath_project_root


# Ensure the script runs from the project root
os.chdir(utils.get_abspath_project_root())


def count_params(tree):
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return sum(leaf.size for leaf in leaves)


def run_bnn_task(
    *,
    seed: int,
    steps: int,
    n_particles: int,
    loss_name: str,
    learning_rate: float,
    width_size: int,
    show_progress: bool,
):
    """Runs a BNN task using a specified loss function and logs the results.

    Parameters:
        seed (int): Random seed for reproducibility.
        steps (int): Number of optimization steps.
        n_particles (int): Number of particles for the loss.
        learning_rate (float): Learning rate for optimization.
        width_size (int): Width of the Bayesian MLP layers.
        save_n_samples (int): Number of samples to save for evaluation.
        show_progress (bool): If True, displays progress during training.
    """
    key = jr.key(seed)
    key, subkey = jr.split(key)

    # Load task and data
    task = bnn_tasks.BNNRegressionProblem()

    data = task.get_data(subkey)

    # Define the prior and Bayesian MLP
    additive_mlp_kwargs = {
        "width_size": width_size,
        "depth": 1,
        "key": subkey,
        "use_final_bias": False,
        "covariate_dim": task.dim,
    }
    prior = bnn.AdditiveBayesianMLP(
        param_to_distribution=lambda p: Laplace(0, jnp.ones_like(p)),
        **additive_mlp_kwargs,
    )

    abmlp = bnn.AdditiveBayesianMLP(
        param_to_distribution=lambda p: Normal(p, 0.02),
        **additive_mlp_kwargs,
    )

    # Get the loss function
    loss_fn = get_losses(prior=prior, k=n_particles)[loss_name]

    # Train the Bayesian MLP
    key, subkey = jr.split(key)
    fitted_abmlp, val_log_likelihoods = train_bnn(
        key,
        abmlp,
        data=data,
        steps=steps,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        show_progress=show_progress,
    )

    # Compute metrics
    key, subkey = jr.split(key)
    computed_metrics = bnn_metrics.compute_metrics(
        subkey,
        fitted_abmlp,
        ground_truth_fn=task.underlying_fn,
        data=data,
    )
    computed_metrics["theta_dim"] = count_params(prior) / 2

    # Save results
    results_path = str(get_abspath_project_root() / "results")
    computed_metrics["val_log_likelihood"] = jnp.asarray(val_log_likelihoods)
    results_name = f"{results_path}/bnn_results/seed={seed}_loss_name={loss_name}_width_size={width_size}_learning_rate={learning_rate}"
    eqx.tree_serialise_leaves(f"{results_name}.eqx", fitted_abmlp)
    jnp.savez(f"{results_name}.npz", **computed_metrics)
    return fitted_abmlp, computed_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a BNN task.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--steps", type=int, help="Training steps.")
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--loss-name", type=str, help="Loss name (see get_losses).")
    parser.add_argument("--width-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    run_bnn_task(
        seed=args.seed,
        steps=args.steps,
        n_particles=args.n_particles,
        learning_rate=args.learning_rate,
        width_size=args.width_size,
        loss_name=args.loss_name,
        show_progress=args.show_progress,
    )
