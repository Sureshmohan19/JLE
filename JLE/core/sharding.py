"""JLE.core.sharding"""

from __future__ import annotations

from collections.abc import Sequence, Mapping

from JLE.core.utils import use_cpp_class
from JLE.core.jaxlib import xla_client as xc
from jaxlib import utils as jaxutils
from JLE.core import jle_backend_connector as jlebc
from JLE.core.xla_op_shardings import (
    are_hlo_sharding_equal, num_ways_dim_sharded, 
    is_hlo_sharding_replicated, shard_to_indices)

safe_zip = jaxutils.safe_zip

Shape = tuple[int, ...]
Device = xc.Device
Index = tuple[slice, ...]
XLADeviceAssignment = Sequence[Device]

# -------- Helper functions -------- #
def _mapping_addressable_devices_indices(
        sharding: JLESharding, 
        global_shape: Shape
) -> Mapping[Device, Index | None]: #type: ignore
    gmap = sharding.mapping_common_devices_indices(global_shape)
    if sharding.is_fully_addressable:
        return gmap
    return {d: gmap[d] for d in sharding._internal_device_list.addressable_device_list} #type: ignore

def _mapping_common_devices_indices(
        sharding: JLESharding,
        global_shape: Shape
) -> Mapping[Device | Index]: #type: ignore
    """
    Returns a mapping from devices to array slices (indices) for a given sharding.
    # 1. Validates that the sharding is compatible with the global array shape.
    # 2. Converts the Sharding object to an XLA HloSharding representation.
    # 3. Checks for unreduced sharding, which is not supported here.
    # 4. Computes the per-device slices using op_sharding_to_indices.
    # 5. Returns a dictionary mapping each device to its corresponding slice.
    """
    sharding.shape_of_shard(global_shape)
    hlo_sharding = sharding._to_xla_hlo_sharding(ndim=len(global_shape))
    if (xc.OpSharding.Type.UNREDUCED in sharding.subgroup_types() or 
        sharding.is_unreduced()):
        raise NotImplementedError(
            f"mapping_common_devices_indices doesn't work with unreduced unfortunately")
    indices = shard_to_indices(sharding, global_shape, len(sharding._device_assignment))
    return dict(safe_zip(sharding._device_assignment, indices))

def _shape_of_shard(self, global_shape: Shape) -> Shape:
    hlo_sharding = self._to_xla_hlo_sharding(len(global_shape))
    if is_hlo_sharding_replicated(hlo_sharding):
        return global_shape
    if hlo_sharding.is_unreduced():
     return global_shape
    
    # Compute the shape of each shard along every dimension for a given global array.
    # 1. Use `num_ways_dim_sharded` to get the number of partitions per axis.
    # 2. Ensure that the number of partitions matches the number of array dimensions.
    # 3. For each dimension, divide the global size by the number of partitions:
    #    - If the division isn’t exact, raise an informative ValueError.
    # 4. Collect the resulting shard sizes for all dimensions and return as a tuple.
    partitions, _ = num_ways_dim_sharded(hlo_sharding)
    assert len(partitions) == len(global_shape), (len(partitions), len(global_shape))
    out = []
    for dim, (s, p) in enumerate(safe_zip(global_shape, partitions)):
        try:
            quotient, remainder = divmod(s, p)
        except TypeError:
            raise NotImplementedError
        if remainder != 0:
            raise ValueError(
                f"Sharding {self} implies that array axis {dim} is partitioned "
                f"{p} times, but the dimension size is {s} "
                f"(full shape: {global_shape}, "
                f"per-dimension tiling factors: {partitions} should evenly divide "
                "the shape)")
        out.append(quotient)
    return tuple(out)

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
    
    def _to_xla_hlo_sharding(self, ndim: int) -> xc.HloSharding:
        raise NotImplementedError("Subclass must override this method")
    
    def with_memory_kind(self, kind: str) -> JLESharding:
        raise NotImplementedError("Subclass must override this method")
    
    # -------- Default methods -------- #
    @property
    def _is_concrete(self) -> bool:
        return True
    
    def addressable_devices(self) -> set[Device]: #type: ignore
        if jlebc.process_count() == 1:
            return self.device_set
        return {d for d in self.device_set if d.process_index == d.client.process_index()}
    
    def mapping_addressable_devices_indices(
            self, 
            global_shape: Shape
    ) -> Mapping[Device, Index | None]: #type: ignore
        return _mapping_addressable_devices_indices(self, global_shape)
    
    def mapping_common_devices_indices(self, global_shape: Shape) -> Mapping[Device, Index]: #type: ignore
        return _mapping_common_devices_indices(self, global_shape)
    
    @property
    def has_addressable_devices(self) -> bool:
        return len(self._internal_device_list.addressable_device_list) > 0
    
    def _addressable_device_assignment(self) -> XLADeviceAssignment:
        if self.is_fully_addressable:
            return self._device_assignment
        return tuple(self._internal_device_list.addressable_device_list) #type: ignore
    
    def shape_of_shard(self, global_shape: Shape) -> Shape:
        return _shape_of_shard(self, global_shape)
    
    def are_shardings_equal(self: JLESharding, other: JLESharding, ndim: int) -> bool:
        try:
            return (
                are_hlo_sharding_equal(self._to_xla_hlo_sharding(ndim), other._to_xla_hlo_sharding(ndim))
                and self._internal_device_list == other._internal_device_list
                and self.memory_kind == other.memory_kind)
        except NotImplementedError:
            return self == other