from importlib.metadata import version

from jaxtyping import install_import_hook

install_import_hook(["flowjax", "cpe_validation", "cpe"], "beartype.beartype")


__version__ = version("cpe_validation")
__all__ = []
