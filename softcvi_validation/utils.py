"""Utility functions."""

from pathlib import Path


def get_palette():
    """Get the palettes used in the plotting of results."""
    return {
        "True": "#00B945",
        "SoftCVI(a=0)": "#FF2C00",
        "SoftCVI(a=0.75)": "#0C5DA5",
        "SoftCVI(a=1)": "#FF9500",
        "SNIS-fKL": "#845B97",
        "ELBO": "#545454",
    }


def get_abspath_project_root():
    """Get the path to the projects root directory."""
    return Path(__file__).parent.parent
