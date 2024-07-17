from importlib.metadata import version

from jaxtyping import install_import_hook

install_import_hook(["flowjax", "softcvi_validation", "softcvi"], "beartype.beartype")


__version__ = version("softcvi_validation")
__all__ = []
