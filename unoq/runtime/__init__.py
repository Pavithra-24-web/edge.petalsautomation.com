"""UNO Q runtime package — import UnoQRuntime for programmatic use."""
from .runtime_backend import UnoQRuntime
from .package_loader import (
    _unpack_pxe,
    _unpack_pe,
    _detect_package_format,
    _install_package,
    _activate_package,
    _load_active_package,
    _load_inference_module,
)

__all__ = [
    "UnoQRuntime",
    "_unpack_pxe",
    "_unpack_pe",
    "_detect_package_format",
    "_install_package",
    "_activate_package",
    "_load_active_package",
    "_load_inference_module",
]
