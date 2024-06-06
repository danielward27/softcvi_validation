from cpe_validation.tasks.eight_schools import EightSchoolsTask
from cpe_validation.tasks.linear_regression import LinearRegressionTask
from cpe_validation.tasks.multimodal_gaussian import (
    MultimodelGaussianFlexibleTask,
    MultimodelGaussianInflexibleTask,
)
from cpe_validation.tasks.slcp import SLCPTask


def get_available_tasks():
    task_list = [
        LinearRegressionTask,
        EightSchoolsTask,
        SLCPTask,
        MultimodelGaussianFlexibleTask,
        MultimodelGaussianInflexibleTask,
    ]
    return {t.name: t for t in task_list}
