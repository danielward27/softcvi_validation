from abc import abstractmethod
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
from softcvi.models import AbstractGuide, AbstractModel
from softcvi.numpyro_utils import (
    validate_data_and_model_match,
)

from softcvi_validation.utils import get_abspath_project_root


class AbstractTask(eqx.Module):
    """A model, guide and method for generating ground truth samples."""

    model: eqx.AbstractVar[AbstractModel]
    guide: eqx.AbstractVar[AbstractGuide]
    name: eqx.AbstractClassVar[str]
    # TODO support auxilary variables?
    # def __check_init__(self):
    #     model = self.model.reparam()
    #     model_trace = shape_only_trace(model)
    #     obs = {}
    #     for name in model.observed_names:
    #         if name not in model_trace:
    #             raise ValueError(
    #                 f"Trace of model does not include observed node {name}.",
    #             )
    #         obs[name] = jnp.empty(
    #             shape=model_trace[name]["value"].shape,
    #             dtype=model_trace[name]["value"].dtype,
    #         )  # keep runtime type checker happy

    #     check_model_guide_match(
    #         model_trace=shape_only_trace(model, obs=obs),
    #         guide_trace=shape_only_trace(self.guide),
    #     )

    @abstractmethod
    def get_latents_and_observed(
        self,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        """Get the observations and parameters.

        The parameters are from a reference posterior if available, otherwise, they are
        the ground truth parameters used to generate the observation.
        """

    def get_latents_and_observed_and_validate(self, key: PRNGKeyArray):
        """Get data and checks matches model trace and model.observed_names."""
        latents, obs = self.get_latents_and_observed(key)
        self._validate_data(latents, obs)

        data = {"latents": latents, "observed": obs}
        for key, dat in data.items():
            for name, arr in dat.items():
                dat[name] = eqx.error_if(
                    x=arr,
                    pred=~jnp.isfinite(arr),
                    msg=f"{name} in {key} had non-finite values",
                )
        return data["latents"], data["observed"]

    def _validate_data(self, latents: dict[str, Array], obs: dict[str, Array]):
        """Validate data matches model with a batch dimension in latents."""
        model = self.model.reparam(set_val=False)
        validate_data_and_model_match(
            obs,
            model,
            assert_present=model.observed_names,
            obs=obs,
        )
        validate_latents_fn = partial(
            validate_data_and_model_match,
            model=model,
            assert_present=model.latent_names,
            obs=obs,
        )
        eqx.filter_vmap(validate_latents_fn)(latents)


class AbstractTaskWithFileReference(AbstractTask):
    """Task with a corresponding reference posterior in reference_posteriors."""

    def get_latents_and_observed(cls, key: PRNGKeyArray):
        """Loads the observation and posteriors from reference_posteriors directory.

        Note this chooses a random observation from the available observations.
        """
        ref_dir = get_abspath_project_root() / "reference_posteriors" / cls.name
        obs = jnp.load(f"{ref_dir}/observations.npz")
        latents = jnp.load(f"{ref_dir}/latents.npz")
        n_obs = list(obs.values())[0].shape[0]
        obs_id = jr.randint(key, (), 0, n_obs)
        latents, obs = (
            {k: jnp.asarray(v)[obs_id] for k, v in d.items()} for d in (latents, obs)
        )
        return latents, obs


class AbstractTaskWithoutReference(AbstractTask):
    """A task without a reference posterior.

    The observation and parameters are generated by sampling the model. For consistency
    with tasks with reference posteriors we add a leading batch dimension of size 1 to
    the latent variables.
    """

    def get_latents_and_observed(self, key: PRNGKeyArray):
        """Generate an observation and ground truth latents from the model."""
        latents = self.model.reparam(set_val=False).sample_joint(key)
        obs = {k: latents.pop(k) for k in self.model.observed_names}
        return {k: v[jnp.newaxis, ...] for k, v in latents.items()}, obs
