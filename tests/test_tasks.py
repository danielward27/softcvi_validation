import jax.random as jr
import pytest
from cnpe.numpyro_utils import shape_only_trace
from numpyro.util import check_model_guide_match

from cnpe_validation.tasks.eight_schools import EightSchoolsTask
from cnpe_validation.tasks.multimodal_gaussian import (
    MultimodelGaussianFlexibleTask,
    MultimodelGaussianInflexibleTask,
)
from cnpe_validation.tasks.sirsde import SIRSDETask
from cnpe_validation.tasks.slcp import SLCPTask

test_cases = [
    EightSchoolsTask,
    MultimodelGaussianFlexibleTask,
    MultimodelGaussianInflexibleTask,
    SIRSDETask,
    SLCPTask,
]


@pytest.mark.parametrize("task", test_cases)
def test_tasks(task):
    key = jr.PRNGKey(0)
    task = task(key)
    _, obs = task.get_latents_and_observed_and_validate(key)

    check_model_guide_match(
        model_trace=shape_only_trace(task.model.reparam(set_val=True), obs=obs),
        guide_trace=shape_only_trace(task.guide, obs=obs),
    )
