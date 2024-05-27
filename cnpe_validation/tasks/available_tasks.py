from cnpe_validation.tasks.eight_schools import EightSchoolsTask
from cnpe_validation.tasks.multimodal_gaussian import (
    MultimodelGaussianFlexibleTask,
    MultimodelGaussianInflexibleTask,
)
from cnpe_validation.tasks.sirsde import SIRSDETask
from cnpe_validation.tasks.slcp import SLCPTask


def get_available_tasks():
    task_list = [
        SIRSDETask,
        EightSchoolsTask,
        SLCPTask,
        MultimodelGaussianInflexibleTask,
        MultimodelGaussianFlexibleTask,
    ]
    return {t.name: t for t in task_list}
