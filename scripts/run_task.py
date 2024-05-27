import argparse
import os

import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import optax

with jaxtyping.install_import_hook(["cnpe", "cnpe_validation"], "beartype.beartype"):
    from cnpe import losses
    from cnpe.train import train

    from cnpe_validation import utils


from cnpe.models import AbstractNumpyroGuide
from jaxtyping import Array, PRNGKeyArray

from cnpe_validation import metrics
from cnpe_validation.tasks.available_tasks import get_available_tasks
from cnpe_validation.tasks.tasks import AbstractTask

os.chdir(utils.get_abspath_project_root())


def main(
    *,
    seed: int,
    task_name: str,
    maximum_likelihood_steps: int,
    contrastive_steps: int,
    num_contrastive: int,
):

    key, subkey = jr.split(jr.PRNGKey(seed))
    task = get_available_tasks()[task_name](subkey)

    key, subkey = jr.split(key)
    true_latents, obs = task.get_latents_and_observed_and_validate(subkey)
    posteriors = {}
    loss_vals = {}

    optimizer = optax.apply_if_finite(
        optax.adam(optax.linear_schedule(1e-3, 1e-4, maximum_likelihood_steps)),
        max_consecutive_errors=100,
    )
    # Train using VI
    method_name = "ELBO"
    loss = losses.NegativeEvidenceLowerBound(task.model.reparam(set_val=True), obs=obs)

    key, subkey = jr.split(key)
    posteriors[method_name], loss_vals[method_name] = train(
        subkey,
        guide=task.guide,
        loss_fn=loss,
        steps=maximum_likelihood_steps,
        optimizer=optimizer,
    )

    # Train using maximum likelihood
    method_name = "Maximum likelihood"
    loss = losses.AmortizedMaximumLikelihood(task.model.reparam(set_val=True))

    posteriors[method_name], loss_vals[method_name] = train(
        subkey,
        guide=task.guide,
        loss_fn=loss,
        steps=maximum_likelihood_steps,
        optimizer=optimizer,
    )

    # Fine tune with contrastive loss
    for stop_grad in [False, True]:
        method_name = f"NPE-PP (stop grad={stop_grad})"

        loss = losses.ContrastiveLoss(
            model=task.model.reparam(),
            obs=obs,
            n_contrastive=num_contrastive,
            stop_grad_for_contrastive_sampling=stop_grad,
        )

        optimizer = optax.apply_if_finite(
            optax.adam(optax.linear_schedule(1e-4, 1e-5, contrastive_steps)),
            max_consecutive_errors=100,
        )
        key, subkey = jr.split(key)
        posteriors[method_name], loss_vals[method_name] = train(
            subkey,
            guide=posteriors["Maximum likelihood"],
            loss_fn=loss,
            steps=contrastive_steps,
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


def compute_metrics(
    key: PRNGKeyArray,
    *,
    task: AbstractTask,
    guide: AbstractNumpyroGuide,
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
    parser = argparse.ArgumentParser(description="NPE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--maximum-likelihood-steps", type=int, default=100)
    parser.add_argument("--contrastive-steps", type=int, default=100)
    parser.add_argument("--num-contrastive", type=int, default=100)
    args = parser.parse_args()

    main(
        seed=args.seed,
        task_name=args.task_name,
        maximum_likelihood_steps=args.maximum_likelihood_steps,
        contrastive_steps=args.contrastive_steps,
        num_contrastive=args.num_contrastive,
    )
