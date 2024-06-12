import os

import jax.numpy as jnp

from scripts.run_task import run_task
from softce_validation import utils
from softce_validation.tasks.available_tasks import get_available_tasks


def main():

    os.chdir(utils.get_abspath_project_root())
    task_names = list(get_available_tasks().keys())

    for task in task_names:
        samples = run_task(
            seed=0,
            task_name=task,
            steps=10,
            return_samples_only=True,
        )
        jnp.savez(f"results/samples/{task}.npz", **samples)


if __name__ == "__main__":
    main()
