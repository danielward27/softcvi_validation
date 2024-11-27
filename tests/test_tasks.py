import jax.random as jr
import pytest
from numpyro.util import check_model_guide_match
from pyrox.numpyro_utils import shape_only_trace

from softcvi_validation.tasks.available_tasks import get_available_tasks


@pytest.mark.parametrize("task", get_available_tasks().values())
def test_tasks(task):
    key = jr.key(0)
    task = task(key)
    _, obs = task.get_latents_and_observed_and_validate(key)

    check_model_guide_match(
        model_trace=shape_only_trace(task.model, obs=obs),
        guide_trace=shape_only_trace(task.guide),
    )
