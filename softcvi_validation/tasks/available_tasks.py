"""Module contains a utility for getting the available tasks."""

from softcvi_validation.tasks.eight_schools import EightSchoolsTask
from softcvi_validation.tasks.garch import GARCHTask
from softcvi_validation.tasks.linear_regression import LinearRegressionTask
from softcvi_validation.tasks.slcp import SLCPTask


def get_available_tasks():
    """Get a list of the available tasks."""
    task_list = [
        GARCHTask,
        EightSchoolsTask,
        LinearRegressionTask,
        SLCPTask,
    ]
    return {t.name: t for t in task_list}
