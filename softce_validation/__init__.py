from importlib.metadata import version

from jaxtyping import install_import_hook

install_import_hook(["flowjax", "softce_validation", "softce"], "beartype.beartype")


__version__ = version("softce_validation")
__all__ = []
