from pathlib import Path


def get_abspath_project_root():
    return Path(__file__).parent.parent
