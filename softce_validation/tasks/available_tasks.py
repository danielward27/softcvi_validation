from softce_validation.tasks.eight_schools import EightSchoolsTask
from softce_validation.tasks.linear_regression import LinearRegressionTask
from softce_validation.tasks.multimodal_gaussian import (
    MultimodelGaussianFlexibleTask,
    MultimodelGaussianInflexibleTask,
)
from softce_validation.tasks.slcp import SLCPTask


def get_available_tasks():
    task_list = [
        LinearRegressionTask,
        EightSchoolsTask,
        SLCPTask,
        MultimodelGaussianFlexibleTask,
        MultimodelGaussianInflexibleTask,
    ]
    return {t.name: t for t in task_list}
