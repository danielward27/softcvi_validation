"""A collection of tasks for validating model performance."""

from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from diffrax import (
    ControlTerm,
    MultiTerm,
    ODETerm,
    SaveAt,
    Tsit5,
    VirtualBrownianTree,
    diffeqsolve,
)
from flowjax.bijections import (
    Affine,
    Chain,
    Stack,
)
from flowjax.distributions import AbstractDistribution, LogNormal, Normal, Transformed
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from gnpe.models import LocScaleHierarchicalModel
from jax import Array, vmap
from numpyro.infer import Predictive

from gnpe_experiments import constraints
from gnpe_experiments.utils import drop_nan_and_warn


def get_surrogate_path():
    return Path(__file__).parent / "surrogate"


def get_surrogate():
    """Get the surrogate likelihood."""
    like = get_sir_surrogate_untrained()
    return eqx.tree_deserialise_leaves(f"{get_surrogate_path()}/surrogate.eqx", like)


def get_processors():
    """The reparameterization used when fitting the surrogate."""
    sirsde = SIRSDE()
    z, x = (jr.uniform(jr.PRNGKey(0), (2, dim)) for dim in [sirsde.z_dim, sirsde.x_dim])
    like = sirsde.infer_processors(z, x)
    return eqx.tree_deserialise_leaves(
        f"{get_surrogate_path()}/processor.eqx",
        like=like,
    )


def get_hierarchical_sir_model(n_obs: int):
    """Get the hierarchical sir model, with the likelihood replaced with a surrogate.

    Note that z and x are not on the easily interpretable scale in this model.
    But we can use the inverse of the transformations returned by get_sir_processors to
    return to the interpretable scales.
    """
    processors = get_processors()
    surrogate = get_surrogate()
    hyperparams = get_sir_model_hyperparams()

    # As the surrogate is trained on a transformed space, we must respect that here
    return LocScaleHierarchicalModel(
        loc=hyperparams["loc"],
        scale=hyperparams["scale"],
        prior=lambda loc, scale: Transformed(
            hyperparams["prior"](loc, scale),
            processors["z"],
        ),
        likelihood=surrogate,
        n_obs=n_obs,
    )


class AbstractSimulator(eqx.Module):
    """Abstract class representing a simulator."""

    z_names: eqx.AbstractVar[tuple[str]]
    z_constraints: eqx.AbstractVar[tuple[constraints.AbstractConstraint]]

    x_names: eqx.AbstractVar[tuple[str]]
    x_constraints: eqx.AbstractVar[tuple[constraints.AbstractConstraint]]

    def __check_init__(self):
        assert len(self.z_names) == len(self.z_constraints)
        assert len(self.x_names) == len(self.x_constraints)

    @abstractmethod
    def simulate(self, key: Array, z: Array):
        """Cary out a single simulation."""
        pass

    @abstractmethod
    def summarize(self, simulation, key: Array):
        """Summarize a single simulation.

        Note the key is used as sometimes it may be desirable e.g. for
        dequantization of x.
        """
        pass

    def to_distribution(self):
        """Returns the simulator, wrapped into a flowjax distribution."""
        return _SimulatorDistribution(self)

    @classmethod
    def infer_processors(cls, z, x):
        """Infer preprocessors.

        Map to unbounded domain, then perform standard scaling.
        """
        # TODO better document and shape checking! Shape won't match all models.

        samps_and_constraints = {
            "z": (z, cls.z_constraints),
            "x": (x, cls.x_constraints),
        }

        preprocessors = {}
        for k, (samps, cons) in samps_and_constraints.items():
            bijection = Stack([c.bijection for c in cons])
            unbounded = jax.vmap(bijection.transform)(samps)
            mean, std = unbounded.mean(axis=0), unbounded.std(axis=0)
            scaler = Affine(-mean / std, 1 / std)
            preprocessors[k] = Chain([bijection, scaler])

        return preprocessors

    @property
    def z_dim(cls):
        return len(cls.z_names)

    @property
    def x_dim(cls):
        return len(cls.x_names)


class _SimulatorDistribution(AbstractDistribution):
    simulator: AbstractSimulator

    def _log_prob(self, *args, **kwargs):
        raise NotImplementedError("Cannot evaluate log prob for simulators.")

    def _sample(self, key, condition):
        sim_key, sum_key = jr.split(key)
        simulation = self.simulator.simulate(sim_key, z=condition)
        return self.simulator.summarize(simulation, sum_key)

    @property
    def shape(self):
        return (self.simulator.x_dim,)

    @property
    def cond_shape(self):
        return (self.simulator.z_dim,)


class SIRSDE(AbstractSimulator):
    """An Susceptible-Infected-Recovered epidemic model, with a stochastic R0."""

    steps: int = 50
    max_solve_steps = 50000
    pop_size: int = 10000
    z_names: ClassVar[tuple[str]] = (
        "infection rate",
        "recovery rate",
        "R0 mean reversion",
        "R0 volatility",
    )
    z_constraints: ClassVar[tuple[constraints.AbstractConstraint]] = (
        constraints.Positive(),
        constraints.Positive(),
        constraints.Positive(),
        constraints.Positive(),
    )

    x_names: ClassVar[tuple[str]] = (
        "mean infections",
        "max infections",
        "median infections",
        "max day",
        "autocorrelation",
    )
    x_constraints: ClassVar[tuple[constraints.AbstractConstraint]] = (
        constraints.Positive(),
        constraints.Positive(),
        constraints.Positive(),
        constraints.Positive(),
        constraints.Interval(0, 1),  # can be negative but in practice is not
    )

    @eqx.filter_jit
    def simulate(self, key: Array, z: Array):
        assert z.shape[-1] == 4
        z = jnp.clip(z, None, 1)  # Rarely lognormal prior could exceed 1
        infection_rate, recovery_rate, r0_mean_reversion, r0_volatility = z.T

        t0, t1 = 0, self.steps + 1
        ode = partial(
            self.ode,
            infection_rate=infection_rate,
            recovery_rate=recovery_rate,
            r0_mean_reversion=r0_mean_reversion,
        )
        sde = partial(self.sde, r0_volatility=r0_volatility)

        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=key)
        r0_init = infection_rate / recovery_rate

        sol = diffeqsolve(
            terms=MultiTerm(ODETerm(ode), ControlTerm(sde, brownian_motion)),
            solver=Tsit5(),
            t0=t0,
            t1=t1,
            dt0=0.01,
            y0=jnp.array([0.99, 0.01, 0, r0_init]),
            saveat=SaveAt(ts=range(1, self.steps + 1)),
            max_steps=self.max_solve_steps,
        )
        return sol.ys[:, 1] * self.pop_size

    def ode(
        self,
        t,
        y,
        *args,
        infection_rate,
        recovery_rate,
        r0_mean_reversion,
    ):
        """ODE portion defined compatible with Diffrax."""
        s, i, r, r0 = y
        newly_infected = r0 * recovery_rate * s
        newly_recovered = recovery_rate * i
        ds = -newly_infected
        di = newly_infected - newly_recovered
        dr = newly_recovered
        dR0 = r0_mean_reversion * (infection_rate / recovery_rate - r0)
        return jnp.hstack((ds, di, dr, dR0))

    def sde(self, t, y, *args, r0_volatility):
        """SDE portion compatible with Diffrax.

        We scale the brownian motion by the square root of R0 to ensure positivity (i.e.
        the mean reversion will dominate).
        """
        scale = jnp.sqrt(jnp.abs(y[-1]))
        return scale * jnp.array([0, 0, 0, r0_volatility])

    def summarize(self, x: Array, key: Array):  # TODO key, x or x, key
        x = jnp.clip(x, a_min=1e-6)
        mean = x.mean()
        max_ = x.max()
        median = jnp.median(x)
        max_at = jnp.argmax(x) + jr.uniform(key)  # Dequantization
        autocorrelation = jnp.clip(_autocorr(x), a_min=0)
        return jnp.array([mean, max_, median, max_at, autocorrelation])


def _autocorr(x, lag=1):
    return jnp.corrcoef(x[:-lag], x[lag:])[0, 1]


def get_sir_model_hyperparams():
    """Return the model hyperparameters used in paper."""
    sirsde = SIRSDE()
    return {
        "loc": Normal(jnp.full(sirsde.z_dim, -2)),
        "scale": LogNormal(-1, scale=jnp.full(sirsde.z_dim, 0.3)),
        "prior": LogNormal,
    }


def get_sir_surrogate_untrained():
    """Get the masked autoregressive flow that acts as a surrogate (untrained)."""
    return masked_autoregressive_flow(
        jr.PRNGKey(0),
        base_dist=Normal(jnp.zeros((SIRSDE().x_dim,))),
        cond_dim=SIRSDE().z_dim,
        invert=False,  # Slower fitting with maximum likelihood, but faster sampling
    )


def main(key, n_sim: int, *, plot_losses: bool = True):
    """Run the SIR model, fit and save the surrogate and the inferred processor."""
    hyperparams = get_sir_model_hyperparams()
    sirsde = SIRSDE()
    model = LocScaleHierarchicalModel(
        **hyperparams,
        likelihood=sirsde.to_distribution(),
        n_obs=1,
    )

    # Simulations for likelihood training
    pred = Predictive(model, num_samples=n_sim)
    key, subkey = jr.split(key)
    simulations = pred(subkey)
    simulations["z"], simulations["x"] = drop_nan_and_warn(
        simulations["z"],
        simulations["x"],
        axis=0,
    )

    simulations["x"], simulations["z"] = (simulations[k].squeeze(1) for k in ["x", "z"])
    processors = sirsde.infer_processors(z=simulations["z"], x=simulations["x"])

    simulations["x"], simulations["z"] = (
        vmap(processors[k].transform)(simulations[k]) for k in ["x", "z"]
    )

    # Learn likelihood on reparameterized space
    key, subkey = jr.split(key)
    surrogate_simulator = get_sir_surrogate_untrained()

    optimizer = optax.apply_if_finite(optax.adam(1e-4), 10)
    key, subkey = jr.split(key)
    surrogate_simulator, losses = fit_to_data(
        key=subkey,
        dist=surrogate_simulator,
        x=simulations["x"],
        condition=simulations["z"],
        max_epochs=600,
        optimizer=optimizer,
        max_patience=10,
    )
    eqx.tree_serialise_leaves(
        f"{get_surrogate_path()}/surrogate.eqx",
        surrogate_simulator,
    )
    eqx.tree_serialise_leaves(f"{get_surrogate_path()}/processor.eqx", processors)
    jnp.savez(f"{get_surrogate_path()}/losses.npz", **losses)

    if plot_losses:
        for k, v in losses.items():
            plt.plot(v, label=k)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # TODO move to scripts folder
    # Run from project root e.g. with: python -m gnpe_experiments.tasks.sirsde.sirsde

    main(key=jr.PRNGKey(0), n_sim=5000, plot_losses=True)
