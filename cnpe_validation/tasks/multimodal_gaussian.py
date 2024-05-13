import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel
from flowjax.distributions import AbstractDistribution, Normal, Uniform, VmapMixture
from flowjax.experimental.numpyro import sample
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from cnpe_validation.tasks.tasks import AbstractTaskWithReference
from cnpe_validation.utils import MLPParameterizedDistribution, TruncNormal


class MultimodalGaussianModel(AbstractNumpyroModel):
    reparameterized: bool | None = None
    observed_names = {"x"}
    reparam_names = {}
    scale = 0.05

    def call_without_reparam(self, obs: dict[str, Float[Scalar, ""]] | None = None):
        mu = sample("mu", Uniform(-1, 1))
        sample(
            "x",
            Normal(jnp.abs(mu), scale=self.scale),
            obs=None if obs is None else obs["x"],
        )

    def get_true_posterior(cls, obs: dict[str, Float[Scalar, ""]]) -> VmapMixture:
        locs = jnp.array([-obs["x"], obs["x"]])
        return VmapMixture(
            eqx.filter_vmap(TruncNormal)(-1, 1, locs, cls.scale),
            weights=jnp.ones(2),
        )


class MultimodalGaussianMisspecifiedGuide(AbstractNumpyroGuide):
    """Guide that is not multimodal - single truncated normal."""

    mu: AbstractDistribution

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        self.mu = MLPParameterizedDistribution(
            key,
            TruncNormal(-1, 1, scale=0.25),
            cond_dim="scalar",
            width_size=10,
        )

    def __call__(self, obs: dict[str, Float[Array, ""]]):
        sample("mu", self.mu, condition=obs["x"])


class MultimodalGaussianWellSpecifiedGuide(AbstractNumpyroGuide):
    """Guide that is multimodal - two truncated normals."""

    mu: AbstractDistribution

    def __init__(
        self,
        key: PRNGKeyArray,
    ):
        truncnorm_mixture = VmapMixture(
            eqx.filter_vmap(TruncNormal)(-1, 1, jnp.zeros(2), 0.25),
            weights=jnp.ones(2),
        )
        self.mu = MLPParameterizedDistribution(
            key,
            truncnorm_mixture,
            cond_dim="scalar",
            width_size=10,
        )

    def __call__(self, obs: dict[str, Float[Array, ""]]):
        sample("mu", self.mu, condition=obs["x"])


class _AbstractMultimodalGaussianTask(AbstractTaskWithReference):

    def get_latents_and_observed(
        self,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        obs_key, posterior_key = jr.split(key)
        _, obs = self.model.reparam(set_val=False).sample_joint(obs_key)

        posterior = self.model.get_true_posterior(obs)
        latents = posterior.sample(posterior_key, (10000,))
        return {"mu": latents}, obs


class MultimodelGaussianMisspecifiedGuideTask(_AbstractMultimodalGaussianTask):
    """A multimodal gaussian task with a misspecified guide.

    The guide is a truncated normal that could only match one mode well.
    """

    model: MultimodalGaussianModel
    guide: MultimodalGaussianMisspecifiedGuide
    name = "multimodal_gaussian_misspecified_guide"

    def __init__(self, key: PRNGKeyArray):
        self.model = MultimodalGaussianModel()
        self.guide = MultimodalGaussianMisspecifiedGuide(key)


class MultimodelGaussianWellSpecifiedGuideTask(_AbstractMultimodalGaussianTask):
    """A multimodal gaussian task with a well specified guide.

    The guide is a mixture of two truncated normal distributions.
    """

    model: MultimodalGaussianModel
    guide: MultimodalGaussianWellSpecifiedGuide
    name = "multimodal_gaussian_well_specified_guide"

    def __init__(self, key: PRNGKeyArray):
        self.model = MultimodalGaussianModel()
        self.guide = MultimodalGaussianWellSpecifiedGuide(key)
