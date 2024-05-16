"""Fits a surrogate SIR model for efficient inference."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from flowjax.train import fit_to_data
from jax import vmap
from numpyro.infer import Predictive

from cnpe_validation.tasks import sirsde
from cnpe_validation.utils import drop_nan_and_warn


def main(key, n_sim: int, *, plot_losses: bool = True):
    """Run the SIR model, fit and save the surrogate and the inferred processor."""
    model = sirsde.SIRSDEModel(n_obs=1, use_surrogate=False)
    model = model.reparam(set_val=False)

    # Simulations for surrogate likelihood training
    pred = Predictive(model, num_samples=n_sim)
    key, subkey = jr.split(key)
    simulations = pred(subkey)
    simulations["x"], simulations["z"] = drop_nan_and_warn(
        *(simulations[k].squeeze(1) for k in ["x", "z"]),
    )

    processors = sirsde.infer_processors(z=simulations["z"], x=simulations["x"])

    simulations["x"], simulations["z"] = (
        vmap(processors[k].transform)(simulations[k]) for k in ["x", "z"]
    )

    # Learn likelihood on reparameterized space
    key, subkey = jr.split(key)
    surrogate_simulator = sirsde.get_surrogate_untrained()

    optimizer = optax.apply_if_finite(
        optax.chain(
            optax.clip_by_global_norm(5),
            optax.adam(optax.linear_schedule(1e-4, 5e-5, 2000)),
        ),
        max_consecutive_errors=100,
    )
    key, subkey = jr.split(key)
    surrogate_simulator, losses = fit_to_data(
        key=subkey,
        dist=surrogate_simulator,
        x=simulations["x"],
        condition=simulations["z"],
        max_epochs=600,
        optimizer=optimizer,
        max_patience=30,
    )
    eqx.tree_serialise_leaves(
        path_or_file=f"{sirsde.get_surrogate_path()}/surrogate.eqx",
        pytree=surrogate_simulator,
    )
    eqx.tree_serialise_leaves(
        path_or_file=f"{sirsde.get_surrogate_path()}/processor.eqx",
        pytree=processors,
    )
    jnp.savez(f"{sirsde.get_surrogate_path()}/losses.npz", **losses)

    if plot_losses:
        for k, v in losses.items():
            plt.plot(v, label=k)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Run from project root e.g. with: python -m scripts.train_sirsde_surrogate
    main(key=jr.PRNGKey(0), n_sim=5000, plot_losses=True)
