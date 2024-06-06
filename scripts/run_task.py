import argparse
import os

import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import optax

with jaxtyping.install_import_hook(["cnpe", "cnpe_validation"], "beartype.beartype"):
    from cpe import losses
    from cpe.train import train

    from cpe_validation import utils


from time import time

import optax
from cpe.models import AbstractGuide
from jaxtyping import Array, PRNGKeyArray

from cpe_validation import metrics
from cpe_validation.tasks.available_tasks import get_available_tasks
from cpe_validation.tasks.tasks import AbstractTask

os.chdir(utils.get_abspath_project_root())


def time_wrapper(fn):
    """Wrap a function to return runtime."""

    def wrapped(*args, **kwargs):
        start = time()
        result = fn(*args, **kwargs)
        end = time()
        return result, end - start

    return wrapped


def main(
    *,
    seed: int,
    task_name: str,
    steps: int,
):
    train_fn = time_wrapper(train)

    key, subkey = jr.split(jr.PRNGKey(seed))
    task = get_available_tasks()[task_name](subkey)

    key, subkey = jr.split(key)
    true_latents, obs = task.get_latents_and_observed_and_validate(subkey)
    posteriors = {}
    loss_vals = {}
    run_times = {}

    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adam(optax.linear_schedule(1e-3, 1e-4, steps)),
        ),
        max_consecutive_errors=100,
    )

    loss_choices = {
        "ELBO": losses.NegativeEvidenceLowerBound(
            model=task.model.reparam(set_val=True),
            obs=obs,
            n_particles=20,
        ),
        "SoftCE": losses.SoftContrastiveEstimationLoss(
            model=task.model.reparam(set_val=True),
            obs=obs,
            n_particles=10,
        ),
    }

    key, subkey = jr.split(key)

    for loss_name, loss in loss_choices.items():

        (posteriors[loss_name], loss_vals[loss_name]), run_times[loss_name] = train_fn(
            subkey,
            guide=task.guide,
            loss_fn=loss,
            steps=steps,
            optimizer=optimizer,
        )

    posterior_metrics = {
        k: compute_metrics(
            key,
            task=task,
            guide=guide,
            obs=obs,
            reference_samples=true_latents,
        )
        for k, guide in posteriors.items()
    }

    results_dir = f"{os.getcwd()}/results/{task.name}/"
    for posterior_name, results in posterior_metrics.items():
        jnp.savez(f"{results_dir}{posterior_name}_{seed}.npz", **results)
        jnp.savez(f"{results_dir}losses_{seed}.npz", **loss_vals)
        jnp.savez(f"{results_dir}run_times_{seed}.npz", **run_times)


def compute_metrics(
    key: PRNGKeyArray,
    *,
    task: AbstractTask,
    guide: AbstractGuide,
    obs: dict,
    reference_samples: dict[str, Array],
):
    key1, key2 = jr.split(key)
    return {
        "mean_log_prob_reference": metrics.mean_log_prob_reference(
            model=task.model,
            guide=guide,
            obs=obs,
            reference_samples=reference_samples,
        ),
        "coverage_probabilities": metrics.coverage_probabilities(
            key1,
            model=task.model,
            guide=guide,
            obs=obs,
            reference_samples=reference_samples,
        ),
        "posterior_mean_l2": metrics.posterior_mean_l2(
            key2,
            task=task,
            guide=guide,
            obs=obs,
            reference_samples=reference_samples,
        ),
    }


if __name__ == "__main__":
    # python -m scripts.run_task --seed=0 --task-name="eight_schools"
    parser = argparse.ArgumentParser(description="SoftCE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()

    main(
        seed=args.seed,
        task_name=args.task_name,
        steps=args.steps,
    )
