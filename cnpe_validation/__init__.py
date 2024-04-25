from importlib.metadata import version

from jaxtyping import install_import_hook

install_import_hook(["flowjax", "cnpe_validation", "cnpe"], "beartype.beartype")


__version__ = version("cnpe_validation")
__all__ = []
