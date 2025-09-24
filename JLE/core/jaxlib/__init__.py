"""JLE.core.jaxlib.init"""

from __future__ import annotations

__all__ = ["xla_client"]

try:
    import jaxlib as jaxlib
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        'JLE requires jaxlib for the backend(for now) and appears to be not installed until this point'
        'Please install jaxlib from https://github.com/jax-ml/jax'
    ) from e

import jaxlib.xla_client as xla_client