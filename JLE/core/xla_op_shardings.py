"""JLE.core.xla_op_shardings"""

from __future__ import annotations

from typing import Sequence, Union
import itertools
import numpy as np

from JLE.core.jaxlib import xla_client as xc

_AXISINDEX = Union[int, slice, tuple[Union[int, slice], ...]]

# -------- Sharding utils deployed in shardings.py -------- #
def is_hlo_sharding_replicated(shard: xc.HloSharding) -> bool:
    """
    Checks whether the given HloSharding represents a fully replicated array 
    (same data on every device).
    """
    return True if shard.num_devices() == 1 else shard.is_replicated()

def are_hlo_sharding_equal(
        shard1: xc.HloSharding,
        shard2: xc.HloSharding) -> bool:
    """
    Compares two HloShardings for equivalence, 
    treating replicated shardings as always equal.
    """
    if shard1 is shard2:
        return True
    if is_hlo_sharding_replicated(shard1) and is_hlo_sharding_replicated(shard2):
        return True
    return shard1 == shard2

def num_ways_dim_sharded(
        shard: xc.HloSharding,
        allow_manual: bool = False) -> tuple[list[int], int]:
    """
    The function num_ways_dim_sharded inspects an xc.HloSharding and returns a 
    tuple (partitions, num_replicas), where partitions is a list of integers 
    showing how many ways each dimension of the array is split across devices, 
    and num_replicas is an integer indicating how many replicas of each shard exist.
    """
    # Disallow fully manual sharding (not supported).
    assert not shard.is_manual()

    # If is_replicated(), the entire array is copied to every device 
    # → no dimension is split, so it returns an empty partition list [] with one replica per shard.
    # If is_unreduced(), it also means "no real sharding was applied" 
    # → same result: ([], 1).
    if shard.is_replicated():
        return [], 1
    if shard.is_unreduced():
        return [], 1
    partitions = shard.tile_assignment_dimensions()
    subgroup_types = shard.subgroup_types()

    if subgroup_types == [xc.OpSharding.Type.REPLICATED]:
        return list(partitions[:-1]), partitions[-1]
    
    elif subgroup_types == [xc.OpSharding.Type.UNREDUCED]:
        return list(partitions[:-1]), 1
    
    elif set(subgroup_types) == {xc.OpSharding.Type.REPLICATED,
                                xc.OpSharding.Type.UNREDUCED}:
        replicated_loc = subgroup_types.index(xc.OpSharding.Type.REPLICATED)
        return list(partitions[:-2]), partitions[-2:][replicated_loc]
    
    elif allow_manual and xc.OpSharding.Type.MANUAL in subgroup_types:
        if subgroup_types == [xc.OpSharding.Type.MANUAL]:
            return list(partitions[:-1]), 1
        else:
            assert (set(subgroup_types) ==
                    {xc.OpSharding.Type.REPLICATED, xc.OpSharding.Type.MANUAL})
            replicated_loc = subgroup_types.index(xc.OpSharding.Type.REPLICATED)
            return list(partitions[:-2]), partitions[-2:][replicated_loc]
        
    elif shard.replicate_on_last_tile_dim():
        return list(partitions[:-1]), partitions[-1]
    else:
        if subgroup_types:
            raise NotImplementedError(f"Unhandled OpSharding type: {shard}. ")
        return list(partitions), 1
    
def shard_to_numpy_indices(
        shard: xc.HloSharding, 
        shape: Sequence[int],
        num_devices: int) -> np.ndarray:
    """
    The function shard_to_numpy_indices translates a sharding specification into a 
    NumPy array of index tuples (slices) that describe exactly which portion of the 
    array each device is responsible for.
    """
    indices = np.empty(num_devices, dtype=np.object_)

    # every device gets the full array
    if is_hlo_sharding_replicated(shard):
        indices.fill((slice(None),) * len(shape))
        return indices
    
    assert num_devices == shard.num_devices()
    partitions, num_replicas = num_ways_dim_sharded(shard=shard)
    assert len(partitions) == len(shape), (len(partitions), len(shape))

    # Compute the NumPy slices for each device based on the sharding.
    # 1. For each array dimension, determine the slices corresponding
    #    to its shards (or full slice if not split).
    # 2. Combine slices across all dimensions to form complete slices
    #    for each shard using a Cartesian product.
    # 3. Assign each slice tuple to the devices according to the 
    #    tile assignment, repeating slices as needed for replicas.
    axis_indices: list[Sequence[_AXISINDEX]] = []
    for dim, n_shards in zip(shape, partitions):
        if n_shards == 1:
            axis_indices.append([slice(None)])
        elif n_shards > 1:
            shard_size, ragged = divmod(dim, n_shards)
            assert not ragged, (dim, n_shards)
            # create slices for this axis, one per shard
            axis_indices.append([slice(i * shard_size, (i + 1) * shard_size)
                            for i in range(n_shards)])
        else:
            raise AssertionError('Unrecognized number of shards. Please file a bug!')

    device_it = iter(shard.tile_assignment_devices())

    # Assign each combination of slices to devices, repeating for replicas
    for idxs in itertools.product(*axis_indices):
        for _ in range(num_replicas):
            indices[next(device_it)] = idxs
    return indices

def shard_to_indices(
        shard: xc.HloSharding,
        shape: Sequence[int],
        num_devices: int) -> tuple[tuple[slice, ...], ...]:
    """
    The function shard_to_indices is just a convenience wrapper around the previous function 
    that flattens those indices into a tuple of slices for easier consumption.
    """
    indices = shard_to_numpy_indices(shard, shape, num_devices)
    return tuple(indices.flat)