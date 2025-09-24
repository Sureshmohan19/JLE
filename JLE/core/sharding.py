"""JLE.core.sharding"""

from __future__ import annotations

from collections.abc import Sequence

from JLE.core.utils import use_cpp_class
from JLE.core.jaxlib import xla_client as xc

Device = xc.Device
XLADeviceAssignment = Sequence[Device]

@use_cpp_class(xc.Sharding)
class JLESharding:
    """which describes how data is laid out across devices"""

    # -------- Abstract methods -------- #
    @property
    def device_set(self) -> Device: #type: ignore
        raise NotImplementedError("Subclass must override this method")
    
    @property
    def is_fully_replicated(self) -> bool:
        raise NotImplementedError("Subclass must override this method")
    
    @property
    def is_fully_addressable(self) -> bool:
        raise NotImplementedError("Subclass must override this method")
    
    @property
    def num_devices(self) -> int:
        raise NotImplementedError("Subclass must override this method")
    
    @property
    def memory_kind(self) -> str | None:
        raise NotImplementedError("Subclass must override this method")
    
    @property
    def _device_assignment(self) -> XLADeviceAssignment:
        raise NotImplementedError("Subclass must override this method")
    
    @property
    def _internal_device_list(self) -> xc.DeviceList: #type: ignore
        raise NotImplementedError("Subclass must override this method")
    
    def with_memory_kind(self, kind: str) -> JLESharding:
        raise NotImplementedError("Subclass must override this method")
    
    # -------- Default methods -------- #
    @property
    def _is_concrete(self) -> bool:
        return True
    
    def addressable_devices(self) -> set[Device]: #type: ignore
        pass
