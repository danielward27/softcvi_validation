"""A collection of tasks for validating model performance."""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel
from diffrax import (
    ControlTerm,
    MultiTerm,
    ODETerm,
    SaveAt,
    Tsit5,
    VirtualBrownianTree,
    diffeqsolve,
)
from flowjax.bijections import Affine, Chain, Stack
from flowjax.distributions import (
    AbstractDistribution,
    LogNormal,
    Normal,
    Transformed,
)
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow
from jax import Array

from cnpe_validation import constraints
from cnpe_validation.tasks.tasks import AbstractTaskWithoutReference


def infer_processors(z, x):
    """Infer preprocessors.

    Map to unbounded domain, then perform standard scaling.
    """
    samps_and_constraints = {
        "z": (z, [constraints.Positive()] * 4),
        "x": (x, [*[constraints.Positive()] * 4, constraints.Interval(0, 1)]),
    }

    preprocessors = {}
    for k, (samps, cons) in samps_and_constraints.items():
        bijection = Stack([c.bijection for c in cons])
        unbounded = jax.vmap(bijection.transform)(samps)
        mean, std = unbounded.mean(axis=0), unbounded.std(axis=0)
        scaler = Affine(-mean / std, 1 / std)
        preprocessors[k] = Chain([bijection, scaler])

    return preprocessors


def get_surrogate_path():
    return Path(__file__).parent / "sirsde_surrogate"


def get_surrogate():
    """Get the trained surrogate likelihood."""
    like = get_surrogate_untrained()
    return eqx.tree_deserialise_leaves(f"{get_surrogate_path()}/surrogate.eqx", like)


def get_surrogate_untrained():
    """Get the masked autoregressive flow that acts as a surrogate (untrained)."""
    return masked_autoregressive_flow(
        jr.PRNGKey(0),
        base_dist=Normal(jnp.zeros(SIRSDESimulator.x_dim)),
        cond_dim=SIRSDESimulator.z_dim,
        invert=False,  # Slower fitting with maximum likelihood, but faster sampling
    )


def get_processors():
    """The reparameterization used when fitting the surrogate."""
    z, x = (
        jr.uniform(jr.PRNGKey(0), (2, dim))
        for dim in [SIRSDESimulator.z_dim, SIRSDESimulator.x_dim]
    )
    like = infer_processors(z, x)
    return eqx.tree_deserialise_leaves(
        f"{get_surrogate_path()}/processor.eqx",
        like=like,
    )


class SIRSDEModel(AbstractNumpyroModel):
    """Hierarchical SIR model, with a hyperprior on the location and scale of p(z).

    Uses a surrogate simulator (by default).

    Args:
        n_obs: The number of observations.
        use_surrogate: Whether to use the surrogate model instead of the simulator.
            Note z and Defaults to True.
    """

    loc: AbstractDistribution
    scale: AbstractDistribution
    z: Callable
    likelihood: AbstractDistribution
    n_obs: int
    reparam_names = {"loc", "scale", "z"}
    observed_names = {"x"}

    def __init__(self, n_obs: int, *, use_surrogate: bool = True):
        if use_surrogate:
            self.likelihood = get_surrogate()
            # As surrogate is trained on scaled z space we transform the prior to match
            processors = get_processors()
            self.z = lambda loc, scale: Transformed(
                LogNormal(loc, scale),
                processors["z"],
            )
        else:
            self.likelihood = SIRSDESimulator()
            self.z = LogNormal

        self.loc = Normal(jnp.full(self.likelihood.cond_shape, -2))
        self.scale = LogNormal(-1, scale=jnp.full(self.likelihood.cond_shape, 0.3))
        self.n_obs = n_obs

    def call_without_reparam(self, obs: dict[str, Array] | None = None):
        """The numpyro model.

        Args:
            obs: The observations. Defaults to None.
        """
        obs = obs["x"] if obs is not None else None

        loc = sample("loc", self.loc)
        scale = sample("scale", self.scale)
        prior = self.z(loc, scale)

        with numpyro.plate("obs", self.n_obs):
            z = sample("z", prior)
            sample("x", self.likelihood, condition=z, obs=obs)


class SIRSDEGuide(AbstractNumpyroGuide):
    """Construct a guide for SIRSDE model with masked autoregressive flows.

    For the loc and scale parameters, the observations are embedded using the
    means and standard deviations across the observations. Note that in the
    reparameterized model, loc, scale and z are independent given the observations.
    """

    loc_base: AbstractDistribution
    scale_base: AbstractDistribution
    z_base: AbstractDistribution
    n_obs: int

    def __init__(self, key: Array, *, n_obs: int, **kwargs):
        keys = jr.split(key, 3)
        self.loc_base = masked_autoregressive_flow(
            key=keys[0],
            base_dist=Normal(jnp.zeros(SIRSDESimulator.z_dim)),
            cond_dim=2 * SIRSDESimulator.x_dim,
            **kwargs,
        )

        self.scale_base = masked_autoregressive_flow(
            key=keys[1],
            base_dist=Normal(jnp.zeros(SIRSDESimulator.z_dim)),
            cond_dim=2 * SIRSDESimulator.x_dim,
            **kwargs,
        )

        self.z_base = masked_autoregressive_flow(
            key=keys[2],
            base_dist=Normal(jnp.zeros((SIRSDESimulator.z_dim,))),
            cond_dim=SIRSDESimulator.x_dim,
            **kwargs,
        )
        self.n_obs = n_obs

    def __call__(self, obs: dict[str, Array]):
        """The numpyro model.

        Args:
            obs: An array of observations.
        """
        obs = obs["x"]
        self._argcheck(obs)  # TODO leave this to beartype
        x_embedding = jnp.concatenate((obs.mean(-2), obs.std(axis=-2)))
        assert x_embedding.ndim == 1
        sample("loc_base", self.loc_base, condition=x_embedding)
        sample("scale_base", self.scale_base, condition=x_embedding)

        with numpyro.plate("obs", obs.shape[-2]):
            z = sample("z_base", self.z_base, condition=obs)

        assert z.shape == (self.n_obs, self.z_base.shape[-1])

    def _argcheck(self, obs):
        if (s := obs.shape[-2]) != self.n_obs:
            raise ValueError(f"Expected obs.shape[-2]=={self.n_obs}, got {s}")


class SIRSDETask(AbstractTaskWithoutReference):
    """Task, using surrogate simulator."""

    model: AbstractNumpyroModel
    guide: AbstractNumpyroGuide
    name = "sirsde"

    def __init__(self, key: Array, n_obs: int = 50):
        self.model = SIRSDEModel(n_obs=n_obs, use_surrogate=True)
        self.guide = SIRSDEGuide(key, n_obs=n_obs)


class SIRSDESimulator(AbstractDistribution):
    """An Susceptible-Infected-Recovered epidemic model, with a stochastic R0.

    This is implemented as a flowjax distribution, with no scaling/reparameterization.
    """

    steps: int = 50
    max_solve_steps = 50000
    pop_size: int = 10000
    z_names: ClassVar[tuple[str]] = (
        "infection rate",
        "recovery rate",
        "R0 mean reversion",
        "R0 volatility",
    )
    x_names: ClassVar[tuple[str]] = (
        "mean infections",
        "max infections",
        "median infections",
        "max day",
        "autocorrelation",
    )
    z_dim: ClassVar[int] = 4
    x_dim: ClassVar[int] = 5

    def _log_prob(self, *args, **kwargs):
        raise NotImplementedError("Cannot evaluate log prob for simulators.")

    def _sample(self, key, condition):
        """Sample the simulator and summarisze the simulation."""
        sim_key, sum_key = jr.split(key)
        simulation = self.simulate(sim_key, condition=condition)
        return self.summarize(simulation, sum_key)

    @eqx.filter_jit
    def simulate(self, key: Array, condition: Array):
        assert condition.shape[-1] == 4
        condition = jnp.clip(
            condition,
            a_min=None,
            a_max=1,
        )  # Rarely lognormal prior could exceed 1
        infection_rate, recovery_rate, r0_mean_reversion, r0_volatility = condition.T

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
        autocorrelation = jnp.clip(jnp.corrcoef(x[:-1], x[1:])[0, 1], a_min=0)
        return jnp.array([mean, max_, median, max_at, autocorrelation])

    @property
    def shape(self):
        return (len(self.x_names),)

    @property
    def cond_shape(self):
        return (len(self.z_names),)
