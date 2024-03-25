# %%
import argparse
import json
import os
from functools import partial

import jax.random as jr
import optax
from flowjax.train import fit_to_variational_target
from gnpe.losses import AmortizedMaximumLikelihood, ContrastiveLoss
from gnpe.models import (
    LocScaleHierarchicalGuide,
)
from gnpe.numpyro_utils import prior_log_density
from gnpe.train import train
from numpyro.infer import Predictive

from gnpe_experiments.metrics import posterior_probability_true
from gnpe_experiments.tasks.sirsde.sirsde import (
    get_hierarchical_sir_model,
)
from gnpe_experiments.utils import get_abspath_project_root

os.chdir(get_abspath_project_root())


def get_sirsde_model_and_guide(key):
    model = get_hierarchical_sir_model(n_obs=50)

    guide = LocScaleHierarchicalGuide(
        key=key,
        z_dim=model.z_dim,
        x_dim=model.x_dim,
        n_obs=model.n_obs,
    )
    return model, guide


TASKS = {"sirsde": get_sirsde_model_and_guide}


def main(
    *,
    seed: int,
    task_name: str,
    maximum_likelihood_steps: int,
    contrastive_steps: int,
    num_contrastive: int,
):
    key, subkey = jr.split(jr.PRNGKey(seed))
    model, guide = TASKS[task_name](subkey)
    posteriors = {}

    # Generate observation from model
    pred = Predictive(model, num_samples=1)
    key, subkey = jr.split(key)
    observations = pred(subkey)
    observations = {k: v.squeeze(0) for k, v in observations.items()}

    key, subkey = jr.split(key)

    # Pretrain using amortized maximum likelihood
    loss = AmortizedMaximumLikelihood(model=model, obs_name="x")

    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(1),
            optax.adam(optax.linear_schedule(1e-2, 5e-4, maximum_likelihood_steps)),
        ),
        max_consecutive_errors=10,
    )

    guide_aml, _ = fit_to_variational_target(
        subkey,
        dist=guide,
        loss_fn=loss,
        steps=maximum_likelihood_steps,
        optimizer=optimizer,
    )
    posteriors["Maximum likelihood"] = guide_aml

    del guide

    # Fine tune with contrastive loss
    for stop_grad in [False, True]:

        loss = ContrastiveLoss(
            model=model,
            obs=observations["x"],
            obs_name="x",
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

        guide_contrastive, _ = train(
            subkey,
            dist=guide_aml,
            loss_fn=loss,
            steps=contrastive_steps,
            optimizer=optimizer,
            convergence_window_size=200,
        )
        posteriors[f"contrastive (stop grad={stop_grad})"] = guide_contrastive

    results = {
        r"$q_{\+\phi}(\theta^*|x_o)$ "
        + k: posterior_probability_true(
            posterior,
            true_latents={k: v for k, v in observations.items() if k != "x"},
            obs=observations["x"],
        )[0].item()
        for k, posterior in posteriors.items()
    }

    results[r"$p(\theta^*)$"] = prior_log_density(
        partial(model, obs=observations["x"]),
        data={k: v for k, v in observations.items() if k != "x"},
        observed_nodes=["x"],
    ).item()

    file_path = f"{os.getcwd()}/results/{task_name}/{seed}.json"

    with open(file_path, "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    # python
    parser = argparse.ArgumentParser(description="NPE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--maximum-likelihood-steps", type=int, default=2000)
    parser.add_argument("--contrastive-steps", type=int, default=2000)
    parser.add_argument("--num-contrastive", type=int, default=20)
    args = parser.parse_args()

    main(
        seed=args.seed,
        task_name=args.task_name,
        maximum_likelihood_steps=args.maximum_likelihood_steps,
        contrastive_steps=args.contrastive_steps,
        num_contrastive=args.num_contrastive,
    )
