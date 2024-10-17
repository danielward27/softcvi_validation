"""Abstract task classes."""

from abc import abstractmethod
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
from pyrox.program import AbstractProgram, remove_reparam

from softcvi_validation.utils import get_abspath_project_root


class AbstractTask(eqx.Module):
    """A model, guide and method for generating ground truth samples.

    The model must accept a key word argument obs, with the array of observations.
    """

    model: eqx.AbstractVar[AbstractProgram]
    guide: eqx.AbstractVar[AbstractProgram]
    name: eqx.AbstractClassVar[str]
    learning_rate: eqx.AbstractVar[float]
    observed_name: str
    latent_names: eqx.AbstractVar[frozenset[str]]

    @abstractmethod
    def get_latents_and_observed(
        self,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], Array]:
        """Get the observations and parameters.

        The parameters are from a reference posterior if available, otherwise, they are
        the ground truth parameters used to generate the observation.
        """

    def get_latents_and_observed_and_validate(self, key: PRNGKeyArray, **kwargs):
        """Get data and checks matches model trace and is finite."""
        latents, obs = self.get_latents_and_observed(key)
        validate = partial(self.model.validate_data, **kwargs)
        eqx.filter_vmap(validate)(latents)
        validate({self.observed_name: obs})

        for name, arr in latents.items():
            latents[name] = eqx.error_if(
                x=arr,
                pred=~jnp.isfinite(arr),
                msg=f"{name} in {key} had non-finite values",
            )
        obs = eqx.error_if(obs, ~jnp.isfinite(obs), "obs had non-finite values.")
        return latents, obs


class AbstractTaskWithFileReference(AbstractTask):
    """Task with a corresponding reference posterior in reference_posteriors."""

    def get_latents_and_observed(cls, key: PRNGKeyArray):
        """Loads the observation and posteriors from reference_posteriors directory.

        Note this chooses a random observation from the available observations.
        """
        ref_dir = get_abspath_project_root() / "reference_posteriors" / cls.name
        obs = jnp.load(f"{ref_dir}/observations.npy")
        latents = jnp.load(f"{ref_dir}/latents.npz")
        n_obs = obs.shape[0]
        obs_id = jr.randint(key, (), minval=0, maxval=n_obs)
        latents = {k: v[obs_id] for k, v in latents.items()}
        return latents, obs[obs_id]


class AbstractTaskWithoutReference(AbstractTask):
    """A task without a reference posterior.

    The observation and parameters are generated by sampling the model. For consistency
    with tasks with reference posteriors we add a leading batch dimension of size 1 to
    the latent variables.
    """

    def get_latents_and_observed(self, key: PRNGKeyArray):
        """Generate an observation and ground truth latents from the model."""
        model = remove_reparam(self.model)
        latents = model.sample(key)
        obs = latents.pop(self.observed_name)
        return {k: v[jnp.newaxis, ...] for k, v in latents.items()}, obs
