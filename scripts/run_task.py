"""Contains the script for running the inference algorithms for a task.

From the project root directory, a run can be carried out using:

``python -m scripts.run_task --seed=0 --task-name="eight_schools"``

"""

import argparse
import os
from functools import partial
from time import time

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import optax
from jaxtyping import Array, PRNGKeyArray

with jaxtyping.install_import_hook(
    ["softcvi", "softcvi_validation"],
    "beartype.beartype",
):
    from flowjax.train.variational_fit import fit_to_variational_target
    from softcvi import losses
    from softcvi.models import AbstractGuide, AbstractReparameterizedModel

    from softcvi_validation import metrics, utils
    from softcvi_validation.tasks.available_tasks import get_available_tasks


os.chdir(utils.get_abspath_project_root())


def time_wrapper(fn):
    """Wrap a function to return runtime."""

    def wrapped(*args, **kwargs):
        start = time()
        result = fn(*args, **kwargs)
        end = time()
        return result, end - start

    return wrapped


def get_losses(
    obs: dict,
    n_particles: int,
    negative_distribution: str,
):
    """Get the loss functions under consideration (assumes fixed observation)."""

    loss_choices = {
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

    return {k: partial(loss, obs=obs) for k, loss in loss_choices.items()}


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

    key, subkey = jr.split(jr.PRNGKey(seed))
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

    train_fn = time_wrapper(
        partial(
            fit_to_variational_target,
            steps=steps,
            show_progress=show_progress,
            return_best=False,
            optimizer=optimizer,
        ),
    )

    loss_choices = get_losses(
        obs=obs,
        n_particles=n_particles,
        negative_distribution=negative_distribution,
    )

    key, subkey = jr.split(key)

    for method_name, loss in loss_choices.items():

        # Returns (((model, guide), losses), runtime)
        ((model, posterior), _), run_time = train_fn(
            subkey,
            (task.model.reparam(set_val=True), task.guide),
            loss_fn=loss,
        )

        metrics = compute_metrics(
            key,
            model=model,
            guide=posterior,
            obs=obs,
            reference_samples=true_latents,
        )
        metrics["run_time"] = run_time

        @partial(jax.vmap, in_axes=[0, None, None])
        def sample_posterior(key, posterior, model):
            sample = posterior.sample(key)
            return model.latents_to_original_space(sample, obs=obs)

        postfix = f"_seed={seed}_k={n_particles}_negative={negative_distribution}.npz"

        key, subkey = jr.split(key)
        samples = sample_posterior(jr.split(subkey, save_n_samples), posterior, model)
        jnp.savez(f"{results_dir}/metrics/{method_name}{postfix}", **metrics)
        jnp.savez(f"{results_dir}/samples/{method_name}{postfix}", **samples)

    # Include true samples
    jnp.savez(
        f"{results_dir}/samples/True{postfix}",
        **{k: v[:save_n_samples] for k, v in true_latents.items()},
    )


def compute_metrics(
    key: PRNGKeyArray,
    *,
    model: AbstractReparameterizedModel,
    guide: AbstractGuide,
    obs: dict,
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


if __name__ == "__main__":
    # python -m scripts.run_task --seed=0 --task-name="eight_schools"

    parser = argparse.ArgumentParser(description="softcvi")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--n-particles", type=int, default=8)
    parser.add_argument("--save-n-samples", type=int, default=1000)
    parser.add_argument("--negative-distribution", type=str, default="proposal")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    run_task(
        seed=args.seed,
        task_name=args.task_name,
        steps=args.steps,
        n_particles=args.n_particles,
        save_n_samples=args.save_n_samples,
        negative_distribution=args.negative_distribution,
        show_progress=args.show_progress,
    )
