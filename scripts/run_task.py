import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import optax

with jaxtyping.install_import_hook(
    ["softce", "softce_validation"],
    "beartype.beartype",
):
    from softce import losses

    from softce_validation import utils


from time import time

from flowjax.train.variational_fit import fit_to_variational_target
from jaxtyping import Array, PRNGKeyArray
from softce.models import AbstractGuide

from softce_validation import metrics
from softce_validation.tasks.available_tasks import get_available_tasks
from softce_validation.tasks.tasks import AbstractTask

os.chdir(utils.get_abspath_project_root())


def time_wrapper(fn):
    """Wrap a function to return runtime."""

    def wrapped(*args, **kwargs):
        start = time()
        result = fn(*args, **kwargs)
        end = time()
        return result, end - start

    return wrapped


def run_task(
    *,
    seed: int,
    task_name: str,
    steps: int,
    n_particles: int,
    save_n_samples: int,
    show_progress: bool,
):
    results_dir = f"{os.getcwd()}/results/{task_name}"

    key, subkey = jr.split(jr.PRNGKey(seed))
    task = get_available_tasks()[task_name](subkey)

    key, subkey = jr.split(key)
    true_latents, obs = task.get_latents_and_observed_and_validate(subkey)

    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(1e-3),
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

    loss_kwargs = {
        "model": task.model.reparam(set_val=True),
        "obs": obs,
        "n_particles": n_particles,
    }

    loss_choices = {
        "SoftCE(a=0)": losses.SoftContrastiveEstimationLoss(**loss_kwargs, alpha=0),
        "SoftCE(a=0.25)": losses.SoftContrastiveEstimationLoss(
            **loss_kwargs, alpha=0.25
        ),
        "SoftCE(a=0.5)": losses.SoftContrastiveEstimationLoss(**loss_kwargs, alpha=0.5),
        "SoftCE(a=0.75)": losses.SoftContrastiveEstimationLoss(
            **loss_kwargs, alpha=0.75
        ),
        "SoftCE(a=1)": losses.SoftContrastiveEstimationLoss(**loss_kwargs, alpha=1),
        "ELBO": losses.EvidenceLowerBoundLoss(**loss_kwargs),
        "SNIS-FKL": losses.SelfNormImportanceWeightedForwardKLLoss(**loss_kwargs),
    }

    key, subkey = jr.split(key)

    for method_name, loss in loss_choices.items():

        (posterior, _), run_time = train_fn(
            subkey,
            task.guide,
            loss_fn=loss,
        )

        metrics = compute_metrics(
            key,
            task=task,
            guide=posterior,
            obs=obs,
            reference_samples=true_latents,
        )
        metrics["run_time"] = run_time

        @partial(jax.vmap, in_axes=[0, None])
        def sample_posterior(key, posterior):
            sample = posterior.sample(key)
            return task.model.latents_to_original_space(sample, obs=obs)

        key, subkey = jr.split(key)
        samples = sample_posterior(jr.split(subkey, save_n_samples), posterior)
        file_name = f"{method_name}_seed={seed}_k={n_particles}.npz"
        jnp.savez(f"{results_dir}/metrics/{file_name}", **metrics)
        jnp.savez(f"{results_dir}/samples/{file_name}", **samples)

    # Include true samples
    jnp.savez(
        f"{results_dir}/samples/True_seed={seed}_k={n_particles}.npz",
        **{k: v[:save_n_samples] for k, v in true_latents.items()},
    )


def compute_metrics(
    key: PRNGKeyArray,
    *,
    task: AbstractTask,
    guide: AbstractGuide,
    obs: dict,
    reference_samples: dict[str, Array],
):
    key1, key2 = jr.split(key)
    kwargs = {"guide": guide, "obs": obs, "reference_samples": reference_samples}
    return {
        "mean_log_prob_reference": metrics.mean_log_prob_reference(
            model=task.model,
            **kwargs,
        ),
        "coverage_probabilities": metrics.coverage_probabilities(
            key1,
            model=task.model,
            **kwargs,
        ),
        "negative_posterior_mean_l2": metrics.negative_posterior_mean_l2(
            key2,
            task=task,
            **kwargs,
        ),
    }


if __name__ == "__main__":
    # python -m scripts.run_task --seed=0 --task-name="eight_schools"
    parser = argparse.ArgumentParser(description="SoftCE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--save-n-samples", type=int, default=1000)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    run_task(
        seed=args.seed,
        task_name=args.task_name,
        steps=args.steps,
        n_particles=args.n_particles,
        save_n_samples=args.save_n_samples,
        show_progress=args.show_progress,
    )
