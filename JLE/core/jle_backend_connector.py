"""JLE.core.jle_backend_connector"""

from __future__ import annotations

from typing import Callable, Union
import os
import threading
import warnings
import dataclasses

from JLE.core.jaxlib import xla_client
from JLE.core import hardware_utils

XlaBackend = xla_client.Client
_XLA_BACKEND = os.getenv("JLE_XLA_BACKEND", "").lower()
_PLATFORM_NAME = os.getenv("JLE_PLATFORM_NAME", "").lower()

def warning_at_fork():
    warnings.warn(
        "let us be clear - you called os.fork() and it will go through because you are using Unix like system, "
        "but be careful as this is incompatible with multithreaded but we are multithreaded. ",
        RuntimeWarning
    )

_at_fork_handler_installed = False

# First think about the backends
# We are skipping all backend plugin support
# as it's not at all needed at this point

BackendFactory = Callable[[], Union[xla_client.Client, None]]

@dataclasses.dataclass(frozen=True)
class BackendRegistration():
    factory: BackendFactory
    priority: int
    show_failure: bool = True

_backend_factories: dict[str, BackendRegistration] = {}
_backends: dict[str, xla_client.Client] = {}
_backend_errors: dict[str, str] = {}
_backend_lock = threading.Lock()
_defaul_backend: xla_client.Client | None = None

_nonexperimental_plugins: set[str] = {'cuda', 'rocm'}

def register_backend_factory(
        name: str, factory: BackendFactory, *,
        priority: int = 0, show_failure: bool = False,
) -> None:
    # check if the backend already exists and raise if it's
    with _backend_lock:
        if name in _backends:
            raise RuntimeError(f"Backend {name} already initialized!")
        
        # add backend to _backend_factories if not registered already
        _backend_factories[name] = BackendRegistration(
            factory, priority, show_failure)
    
def _init_backend(platform: str) -> xla_client.Client:
    registration = _backend_factories.get(platform, None)
    if registration is None:
        raise RuntimeError(
            f"The provided backend '{platform} is not in the list of known backends: "
            f"{list(_backend_factories.keys())}."
        )
    backend = registration.factory()
    if backend is None:
        raise RuntimeError(
            f"Failed to initialise backend '{platform}"
        )
    if backend.device_count() == 0 and len(backend._get_all_devices()) == 0:
        raise RuntimeError(
            f"Backend '{platform} provides no devices actually!"
        )

    return backend
        
def backends() -> dict[str, xla_client.Client]:
    global _backends
    global _backend_errors
    global _defaul_backend
    global _at_fork_handler_installed

    # return immediately if backend found in _backends
    with _backend_lock:
        if _backends:
            return _backends
    
        # The following code ensures safe behavior when a process forks on Unix systems. 
        # By registering _at_fork to run before every fork, it prevents issues with 
        # inconsistent states in threads, locks, or resources, reducing the risk of 
        # deadlocks or corruption in the child process.
        if not _at_fork_handler_installed and hasattr(os, "register_at_fork"):
            os.register_at_fork(before=warning_at_fork)
            _at_fork_handler_installed = True
        
        # For now, we only check _backend_factories and init backends
        if _backend_factories.items():
            platform_reg = [
                (plat, reg.priority, reg.show_failure)
                for plat, reg in _backend_factories.items()
            ]
        
        default_priority = -999
        for platform, priority, show_failure in platform_reg:
            try:
                if platform == "cuda" and not hardware_utils.has_nvidia_gpu():
                    continue

                backend = _init_backend(platform)
                _backends[platform] = backend

                if priority > default_priority:
                    _defaul_backend = backend
                    default_priority = priority            
            except Exception as e:
                msg = f"Failed to initialise the backend as requested, {platform}: {e}"
                if show_failure:
                    msg += "(set JLE_PLATFORMS= '' to automatically choose an available backend)"
                    msg += "(or set JLE_PLATFORMS=cpu to skip this backend)"
                    raise RuntimeError(msg)
                else:
                    _backend_errors[platform] = str(e)
        
        assert _defaul_backend is not None
        return _backends
    
def canonicalize_platform(platform):
    pass
        
def _get_backend_uncached(platform: xla_client.Client | str | None = None) -> xla_client.Client:
    # if platform is xla_client.Client, just return it
    if platform is not None and not isinstance(platform, str):
        assert isinstance(platform, xla_client.Client)
        return platform
    
    platform = (platform or _XLA_BACKEND or _PLATFORM_NAME or None)

    back = backends()
    if platform is not None:
        platform = canonicalize_platform(platform)
        backend = back.get(platform, None)
        if backend is None:
            if platform in _backend_errors:
                raise RuntimeError(
                    f"{platform} failed to initialise due to the following reason: "
                    f"{_backend_errors[platform]}."
                    f"Available backends are {list(back)}"
                )
            raise RuntimeError(
                f"{platform} is not known. Available backends are {list(back)}"
            )
        return backend
    else:
        assert _defaul_backend is not None
        return _defaul_backend
        
def get_backend(platform: xla_client.Client | str | None = None) -> xla_client.Client:
    return _get_backend_uncached(platform)

# -------- Devices related functions -------- #
def devices(backend: xla_client.Client | str | None = None) -> list[xla_client.Device]:
    return get_backend(backend).devices()

def device_count(backend: xla_client.Client | str | None = None) -> int:
    return int(get_backend(backend).device_count())

def local_device_count(backend: xla_client.Client | str | None = None) -> int:
    return int(get_backend(backend).local_device_count())

def get_device_backend(device: xla_client.Device | None = None) -> xla_client.Client:
    if device is not None:
        return device.Device
    return get_backend()

def default_backend() -> str:
    return get_backend(None).platform

# -------- Process related functions -------- #
def process_index(backend: xla_client.Client | str | None = None) -> int:
    return get_backend(backend).process_index()

def process_count(backend: xla_client.Client | str | None = None) -> int:
    pro = (d.process_index  for d in devices(backend))
    return max(pro, default=0) + 1