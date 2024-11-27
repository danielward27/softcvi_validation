"""Utility functions."""

from pathlib import Path

import matplotlib


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


def darken_color(color):
    """Darken a color for easier visualisation when plotting (e.g. of text)."""
    color = matplotlib.colors.ColorConverter.to_rgb(color)
    return tuple(0.7 * c for c in color)


def get_abspath_project_root():
    """Get the path to the projects root directory."""
    return Path(__file__).parent.parent
