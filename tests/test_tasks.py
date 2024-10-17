import jax.random as jr
import pytest
from numpyro.util import check_model_guide_match
from softcvi.numpyro_utils import shape_only_trace

from softcvi_validation.tasks.available_tasks import get_available_tasks


@pytest.mark.parametrize("task", get_available_tasks().values())
def test_tasks(task):
    key = jr.key(0)
    task = task(key)
    _, obs = task.get_latents_and_observed_and_validate(key)

    check_model_guide_match(
        model_trace=shape_only_trace(task.model, obs=obs),
        guide_trace=shape_only_trace(task.guide),
    )


# %%

import argparse
import os
from functools import partial
from time import time

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import optax
from flowjax.train import fit_to_key_based_loss
from jaxtyping import Array, PRNGKeyArray
from pyrox import losses
from pyrox.program import AbstractProgram, SetKwargs

from softcvi_validation import metrics, utils
from softcvi_validation.tasks.available_tasks import get_available_tasks


def get_losses(
    n_particles: int,
    negative_distribution: str,
):
    """Get the loss functions under consideration."""
    return {
        "SoftCVI(a=0)": losses.SoftContrastiveEstimationLoss(
            n_particles=n_particles,
            alpha=0,
            negative_distribution=negative_distribution,
        ),
        "SoftCVI(a=0.75)": losses.SoftContrastiveEstimationLoss(
            n_particles=n_particles,
            alpha=0.75,
            negative_distribution=negative_distribution,
        ),
        "SoftCVI(a=1)": losses.SoftContrastiveEstimationLoss(
            n_particles=n_particles,
            alpha=1,
            negative_distribution=negative_distribution,
        ),
        "ELBO": losses.EvidenceLowerBoundLoss(n_particles=n_particles),
        "SNIS-fKL": losses.SelfNormImportanceWeightedForwardKLLoss(
            n_particles=n_particles,
        ),
    }


def run_task(
    *,
    seed: int,
    task_name: str,
    steps: int,
    n_particles: int,
    save_n_samples: int,
    negative_distribution: bool,
    show_progress: bool,
):
    """Run the task for each inference method, saving metrics and samples.

    Args:
        seed: Integer seed value.
        task_name: The task name to run (see ``get_available_tasks``).
        steps: Number of optimization steps to take.
        n_particles: Number of particles to use in loss approximation.
        save_n_samples: Number of posterior samples to save.
        negative_distribution: The negative distribution to use for
            ``SoftContrastiveEstimationLoss``.
        show_progress: Whether to show the progress bar.
    """
    results_dir = f"{os.getcwd()}/results/{task_name}"

    key, subkey = jr.split(jr.key(seed))
    task = get_available_tasks()[task_name](subkey)

    key, subkey = jr.split(key)
    true_latents, obs = task.get_latents_and_observed_and_validate(subkey)

    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(task.learning_rate),
        ),
        max_consecutive_errors=100,
    )

    train_fn = partial(
        fit_to_key_based_loss,
        steps=steps,
        show_progress=show_progress,
        optimizer=optimizer,
    )

    loss_choices = get_losses(
        n_particles=n_particles,
        negative_distribution=negative_distribution,
    )

    key, subkey = jr.split(key)

    for method_name, loss in loss_choices.items():

        # Returns (((model, guide), losses), runtime)
        (model, posterior), _ = train_fn(
            subkey,
            (SetKwargs(task.model, obs=obs), task.guide),
            loss_fn=loss,
        )
        model = model.program  # Undo set kwargs

        metrics = compute_metrics(
            key,
            model=model,
            guide=posterior,
            obs=obs,
            reference_samples=true_latents,
        )

        @partial(jax.vmap, in_axes=[0, None, None])
        def sample_posterior(key, posterior, model):
            sample = posterior.sample(key)
            return model.latents_to_original_space(sample, obs=obs)

        postfix = f"_seed={seed}_k={n_particles}_negative={negative_distribution}.npz"

        key, subkey = jr.split(key)
        samples = sample_posterior(jr.split(subkey, save_n_samples), posterior, model)
        # jnp.savez(f"{results_dir}/metrics/{method_name}{postfix}", **metrics)
        # jnp.savez(f"{results_dir}/samples/{method_name}{postfix}", **samples)

    # Include true samples
    # jnp.savez(
    #     f"{results_dir}/samples/True{postfix}",
    #     **{k: v[:save_n_samples] for k, v in true_latents.items()},
    # )


def compute_metrics(
    key: PRNGKeyArray,
    *,
    model: AbstractProgram,
    guide: AbstractProgram,
    obs: Array,
    reference_samples: dict[str, Array],
):
    """Compute the performance metrics."""
    key1, key2 = jr.split(key)
    kwargs = {
        "model": model,
        "guide": guide,
        "obs": obs,
        "reference_samples": reference_samples,
    }

    return {
        "mean_log_prob_reference": metrics.mean_log_prob_reference(
            **kwargs,
        ),
        "coverage_probabilities": metrics.coverage_probabilities(
            key1,
            **kwargs,
        ),
        "negative_posterior_mean_l2": metrics.negative_posterior_mean_l2(
            key2,
            **kwargs,
        ),
    }


# %%


run_task(
    seed=10000,
    task_name="eight_schools",
    steps=10,
    n_particles=2,
    show_progress=True,
    save_n_samples=100,
    negative_distribution="proposal",
)
