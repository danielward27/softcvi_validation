import argparse
import os

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import optax

with jaxtyping.install_import_hook(["cnpe", "cnpe_validation"], "beartype.beartype"):
    from cnpe.losses import (
        AmortizedMaximumLikelihood,
        ContrastiveLoss,
        NegativeEvidenceLowerBound,
    )
    from cnpe.train import train

    from cnpe_validation.tasks.eight_schools import EightSchoolsTask
    from cnpe_validation.tasks.multimodal_gaussian import (
        MultimodelGaussianMisspecifiedGuideTask,
        MultimodelGaussianWellSpecifiedGuideTask,
    )
    from cnpe_validation.tasks.sirsde import SIRSDETask
    from cnpe_validation.tasks.tasks import AbstractTaskWithReference
    from cnpe_validation.tasks.two_moons import TwoMoonsTask
    from cnpe_validation.utils import get_abspath_project_root

os.chdir(get_abspath_project_root())

TASKS = {
    t.name: t
    for t in [
        SIRSDETask,
        EightSchoolsTask,
        TwoMoonsTask,
        MultimodelGaussianMisspecifiedGuideTask,
        MultimodelGaussianWellSpecifiedGuideTask,
    ]
}


def main(
    *,
    seed: int,
    task_name: str,
    maximum_likelihood_steps: int,
    contrastive_steps: int,
    num_contrastive: int,
):

    key, subkey = jr.split(jr.PRNGKey(seed))
    task = TASKS[task_name](subkey)

    key, subkey = jr.split(key)
    obs, true_latents = task.get_latents_and_observed_and_validate(subkey)
    posteriors = {}
    losses = {}

    optimizer = optax.apply_if_finite(
        optax.adam(optax.linear_schedule(1e-3, 1e-4, maximum_likelihood_steps)),
        max_consecutive_errors=100,
    )

    # Train using VI
    method_name = "evidence lower bound"
    loss = NegativeEvidenceLowerBound(task.model.reparam(set_val=True), obs=obs)

    key, subkey = jr.split(key)
    posteriors[method_name], losses[method_name] = train(
        subkey,
        guide=task.guide,
        loss_fn=loss,
        steps=maximum_likelihood_steps,
        optimizer=optimizer,
    )

    # Train using maximum likelihood
    method_name = "maximum likelihood"
    loss = AmortizedMaximumLikelihood(task.model.reparam(set_val=True))

    posteriors[method_name], losses[method_name] = train(
        subkey,
        guide=task.guide,
        loss_fn=loss,
        steps=maximum_likelihood_steps,
        optimizer=optimizer,
    )

    # Fine tune with contrastive loss
    for stop_grad in [False, True]:
        method_name = f"npe-pp (stop grad={stop_grad})"

        loss = ContrastiveLoss(
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
        posteriors[method_name], losses[method_name] = train(
            subkey,
            guide=posteriors["maximum likelihood"],
            loss_fn=loss,
            steps=contrastive_steps,
            optimizer=optimizer,
        )

    def compute_true_latent_prob(true_latents):  # For a single latent
        log_probs = {
            k: posterior.log_prob_original_space(
                latents=true_latents,
                model=task.model,
                obs=obs,
            )
            for k, posterior in posteriors.items()
        }
        log_probs["prior"] = task.model.reparam(set_val=False).prior_log_prob(
            true_latents,
        )
        return log_probs

    if isinstance(task, AbstractTaskWithReference):  # Batch of "true" samples
        compute_true_latent_prob = eqx.filter_vmap(compute_true_latent_prob)

    log_prob_true = compute_true_latent_prob(true_latents)
    results_dir = f"{os.getcwd()}/results/{task.name}/"
    jnp.savez(results_dir + f"true_posterior_log_probs_{seed}.npz", **log_prob_true)
    jnp.savez(results_dir + f"losses_{seed}.npz", **losses)


if __name__ == "__main__":
    # python -m scripts.run_task --seed=0 --task-name="eight_schools"
    parser = argparse.ArgumentParser(description="NPE")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--maximum-likelihood-steps", type=int, default=4000)
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
