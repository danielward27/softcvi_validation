# # %%
# import jax.random as jr
# import pytest
# from cnpe.numpyro_utils import shape_only_trace

# from cnpe_validation.tasks.eight_schools import EightSchoolsTask
# from cnpe_validation.tasks.tasks import AbstractTask


# def validate_task(task: AbstractTask):
#     model_trace = shape_only_trace(task.model)
#     guide_trace =


# test_cases = [
#     EightSchoolsTask(jr.PRNGKey(0)),
# ]


# @pytest.mark.parameterize(("task",), test_cases)
# def test_tasks(task: AbstractTask):
#     validate_task(task)
