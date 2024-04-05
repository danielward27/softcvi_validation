import argparse
import json
import os
from functools import partial

import jax.random as jr
import matplotlib.pyplot as plt
import optax
from cnpe.losses import AmortizedMaximumLikelihood, ContrastiveLoss
from cnpe.numpyro_utils import prior_log_density
from cnpe.train import train

from cnpe_validation.metrics import posterior_probability_true
from cnpe_validation.tasks.eight_schools_non_centered import EightSchoolsNonCenteredTask
from cnpe_validation.tasks.sirsde import SIRSDETask
from cnpe_validation.utils import get_abspath_project_root

os.chdir(get_abspath_project_root())

TASKS = {
    "sirsde": SIRSDETask,
    "eight_schools_non_centered": EightSchoolsNonCenteredTask,
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
    obs, true_latents = task.get_obs_and_parameters(subkey)

    posteriors = {}

    key, subkey = jr.split(key)

    # Pretrain using amortized maximum likelihood
    loss = AmortizedMaximumLikelihood(model=task.model, obs_name=task.obs_name)

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
            obs_name=task.obs_name,
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

    results = {
        r"$q_{\+\phi}(\theta^*|x_o)$ "
        + k: posterior_probability_true(
            posterior,
            true_latents=true_latents,
            obs=obs,
        )[0].item()
        for k, posterior in posteriors.items()
    }

    results[r"$p(\theta^*)$"] = prior_log_density(
        partial(task.model, obs=obs),
        data=true_latents,
        observed_nodes=[task.obs_name],
    ).item()

    file_path = f"{os.getcwd()}/results/{task_name}/{seed}.json"

    with open(file_path, "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    # python -m scripts.run_task --seed=0 --task-name="eight_schools_non_centered"
    parser = argparse.ArgumentParser(description="NPE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--maximum-likelihood-steps", type=int, default=2000)
    parser.add_argument("--contrastive-steps", type=int, default=2000)
    parser.add_argument("--num-contrastive", type=int, default=20)
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
