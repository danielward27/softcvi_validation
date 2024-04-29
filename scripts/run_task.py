import argparse
import os
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import matplotlib.pyplot as plt
import optax

with jaxtyping.install_import_hook(["cnpe", "cnpe_validation"], "beartype.beartype"):
    from cnpe.losses import AmortizedMaximumLikelihood, ContrastiveLoss
    from cnpe.numpyro_utils import prior_log_density
    from cnpe.train import train

    from cnpe_validation.tasks.eight_schools import EightSchoolsTask
    from cnpe_validation.tasks.sirsde import SIRSDETask
    from cnpe_validation.tasks.tasks import AbstractTaskWithReference
    from cnpe_validation.tasks.two_moons import TwoMoonsTask
    from cnpe_validation.utils import get_abspath_project_root

os.chdir(get_abspath_project_root())

TASKS = {
    "sirsde": SIRSDETask,
    "eight_schools": EightSchoolsTask,
    "two_moons": TwoMoonsTask,
}


def main(
    *,
    seed: int,
    task_name: str,
    maximum_likelihood_steps: int,
    contrastive_steps: int,
    num_contrastive: int,
    plot_losses: bool,
):

    key, subkey = jr.split(jr.PRNGKey(seed))
    task = TASKS[task_name](subkey)

    key, subkey = jr.split(key)
    obs, true_latents = task.get_observed_and_latents_and_check(subkey)

    posteriors = {}

    key, subkey = jr.split(key)

    # Pretrain using amortized maximum likelihood
    loss = AmortizedMaximumLikelihood(task.model)

    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(optax.linear_schedule(1e-2, 5e-4, maximum_likelihood_steps)),
        ),
        max_consecutive_errors=10,
    )

    guide_aml, losses = train(
        subkey,
        guide=task.guide,
        loss_fn=loss,
        steps=maximum_likelihood_steps,
        optimizer=optimizer,
    )

    if plot_losses:
        plt.plot(losses)
        plt.show()
    posteriors["Maximum likelihood"] = guide_aml

    # Fine tune with contrastive loss
    for stop_grad in [False, True]:

        loss = ContrastiveLoss(
            model=task.model,
            obs=obs,
            n_contrastive=num_contrastive,
            stop_grad_for_contrastive_sampling=stop_grad,
        )

        key, subkey = jr.split(key)

        optimizer = optax.apply_if_finite(
            optax.chain(
                optax.clip_by_global_norm(1),
                optax.adam(1e-4),
            ),
            max_consecutive_errors=500,
        )

        guide_contrastive, losses = train(
            subkey,
            guide=guide_aml,
            loss_fn=loss,
            steps=contrastive_steps,
            optimizer=optimizer,
        )
        if plot_losses:
            plt.plot(losses)
            plt.show()

        posteriors[f"contrastive (stop grad={stop_grad})"] = guide_contrastive

    def compute_true_latent_prob(true_latents):  # For a single latent
        results = {
            k: posterior.log_prob_original_space(
                latents=true_latents,
                model=task.model,
                obs=obs,
            )
            for k, posterior in posteriors.items()
        }

        results[r"$p(\theta^*)$"] = prior_log_density(
            partial(task.model.call_without_reparam, obs=obs),
            data=true_latents,
            observed_nodes=task.model.observed_names,
        )
        return results

    if isinstance(task, AbstractTaskWithReference):  # Batch of "true" samples
        compute_true_latent_prob = eqx.filter_vmap(compute_true_latent_prob)

    results = compute_true_latent_prob(true_latents)
    file_path = f"{os.getcwd()}/results/{task_name}/{seed}.npz"
    jnp.savez(file_path, **results)


if __name__ == "__main__":
    # python -m scripts.run_task --seed=0 --task-name="eight_schools"
    parser = argparse.ArgumentParser(description="NPE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--maximum-likelihood-steps", type=int, default=2000)
    parser.add_argument("--contrastive-steps", type=int, default=2000)
    parser.add_argument("--num-contrastive", type=int, default=2000)
    parser.add_argument("--plot-losses", action="store_true")
    args = parser.parse_args()

    main(
        seed=args.seed,
        task_name=args.task_name,
        maximum_likelihood_steps=args.maximum_likelihood_steps,
        contrastive_steps=args.contrastive_steps,
        num_contrastive=args.num_contrastive,
        plot_losses=args.plot_losses,
    )
