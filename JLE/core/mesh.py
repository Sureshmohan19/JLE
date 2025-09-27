"""JLE.core.mesh"""

from __future__ import annotations

import enum
import functools
import contextlib
import collections
import threading
import math
from typing import Any, NamedTuple
from collections.abc import Sequence, Hashable

import numpy as np

from JLE.core.jaxlib import xla_client as xc
from JLE.core import jle_backend_connector as jlebc
from JLE.core.utils import safe_zip

MeshAxisName = Any
ResourceAxisName = Hashable
_mesh_object_dict = {}

# -------- Helper Functions -------- #
def all_axis_types_match(axis_types, cm: AxisType) -> bool:
    if not axis_types:
        return False
    return all(t == cm for t in axis_types)

def any_axis_types_match(axis_types, cm: AxisType) -> bool:
    if not axis_types:
        return False
    return any(t == cm for t in axis_types)

def all_axis_auto_or_manual(axis_types) -> bool:
    if not axis_types:
        return False
    return all(t == AxisType.Auto or t == AxisType.Manual for t in axis_types)

def any_axis_auto_or_manual(axis_types) -> bool:
    if not axis_types:
        return False
    return any(t == AxisType.Auto or t == AxisType.Manual for t in axis_types)

def _normalize_axis_types(axis_names, axis_types):
    # first make sure axis_types is not None and axis_names is a tuple
    axis_types = ((AxisType.Auto,) * len(axis_names) if axis_types is None else axis_types)
    if not isinstance(axis_names, tuple):
        axis_names = tuple(axis_names)

    # now just check instances and length
    if not all(isinstance(types, AxisType) for types in axis_types):
        raise ValueError(
            f"axis_types must be an instance of JLE.AxisType but got {axis_types} of type {tuple(type(a) for a in axis_types)}.")
    
    if len(axis_names) != len(axis_types):
        raise ValueError(
            f"Number of axis_types should match the number of axis_types but we got axis_names={len(axis_names)} and axis_types={len(axis_types)}.")
    
    return axis_types

# -------- MeshEnvContext class -------- #
class JLEMeshEnvContext(NamedTuple):
    physical_mesh: JLEMesh

    def with_mesh(self, mesh: JLEMesh):
        new_axes = set(mesh.axis_names)
        current_axes = set(self.physical_mesh.axis_names)
        other_used_axes = self.resource_axes - current_axes
        overlap = new_axes & other_used_axes

        if overlap:
            raise ValueError(
                f"Cannot update the mesh of the current env context. "
                f"The already defined axes are {','.join(sorted(f"'{a}'" for a in overlap))}. ")

        return self._replace(physical_mesh = mesh)
    
    @property
    def physical_resource_axes(self) -> set[ResourceAxisName]:
        return set(self.physical_mesh.axis_names)
    
    @property
    def resource_axes(self) -> set[ResourceAxisName]:
        return self.physical_resource_axes
    
    @property
    def shape(self):
        return self.physical_mesh.shape
    
    @property
    def local_shape(self):
        return self.physical_mesh.local_mesh.shape
    
    def __repr__(self):
        repr_ = ", ".join(f"'{k}': {v}" for k, v in self.physical_mesh.shape.items())
        return f"JLEMeshEnvContext(mesh=JLEMesh({repr_}))"

# -------- AxisType Enum -------- #
class AxisType(enum.Enum):
    Auto = enum.auto()
    Explicit = enum.auto()
    Manual = enum.auto()

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"
    
    def __str__(self):
        return self.name
    
# -------- Base Mesh Class -------- #
class BaseJLEMesh:
    axis_names: tuple[MeshAxisName, ...]
    shape_tuple: tuple[tuple[str, int], ...]
    axis_types: tuple[AxisType, ...]

    @functools.cached_property
    def are_all_axes_manual(self) -> bool:
        return all_axis_types_match(self.axis_types, AxisType.Manual)
    
    @functools.cached_property
    def are_all_axes_auto(self) -> bool:
        return all_axis_types_match(self.axis_types, AxisType.Auto)
    
    @functools.cached_property
    def are_all_axes_explicit(self) -> bool:
        return all_axis_types_match(self.axis_types, AxisType.Explicit)
    
    @functools.cached_property
    def _are_all_axes_auto_or_manual(self) -> bool:
        return all_axis_auto_or_manual(self.axis_types)
    
    @functools.cached_property
    def _any_axis_manual(self) -> bool:
        return any_axis_types_match(self.axis_types, AxisType.Manual)
    
    @functools.cached_property
    def _any_axis_auto(self) -> bool:
        return any_axis_types_match(self.axis_types, AxisType.Auto)
    
    @functools.cached_property
    def _any_axis_explicit(self) -> bool:
        return any_axis_types_match(self.axis_types, AxisType.Explicit)
    
    @functools.cached_property
    def _any_axis_auto_or_manual(self) -> bool:
        return any_axis_auto_or_manual(self.axis_types)
    
    # -------- Get one axes -------- #
    @functools.cached_property
    def auto_axes(self):
        return tuple(name_ for name_, types_ in safe_zip(self.axis_names, self.axis_types) if types_ == AxisType.Auto)
    
    @functools.cached_property
    def manual_axes(self):
        return tuple(name_ for name_, types_ in safe_zip(self.axis_names, self.axis_types) if types_ == AxisType.Manual)
    
    @functools.cached_property
    def explicit_axes(self):
        return tuple(name_ for name_, types_ in safe_zip(self.axis_names, self.axis_types) if types_ == AxisType.Explicit)
    
    @functools.cached_property
    def _name_to_type(self):
        return dict(safe_zip(self.axis_names, self.axis_types))
    
def _unpickle_mesh(devices, axis_names, axis_types):
    return JLEMesh(devices, axis_names, axis_types)
    
# -------- JLE Mesh class -------- #
class JLEMesh(BaseJLEMesh, contextlib.ContextDecorator):
    devices: np.ndarray
    axis_names: tuple[MeshAxisName, ...]

    def __new__(cls, 
                devices: np.ndarray | Sequence[xc.Device],
                axis_names: str | Sequence[MeshAxisName],
                axis_types: tuple[AxisType, ...] | None = None,
    ):
        """
        __new__ is a special method in Python that controls object creation.
        It's called before __init__ and is responsible for actually creating 
        and returning the instance.
        """
        if not isinstance(devices, np.ndarray):
            devices = np.array(devices)
        
        # Make axis_names a tuple
        if isinstance(axis_names, str):
            axis_names = (axis_names,)
        axis_names = tuple(axis_names)

        # Mesh axis names cannot be None
        if any(name is None for name in axis_names):
            raise ValueError(
                f"Mesh axis name cannot be None but you gave {axis_names}")
        
        if devices.ndim != len(axis_names):
            raise ValueError(
                f"devices argument's ndim: {devices.ndim} must be equal to the length of axis_names: {len(axis_names)}.")
        
        axis_types = _normalize_axis_types(axis_names, axis_types)
        key = (axis_names, devices.shape, tuple(devices.flat), axis_types)
        val = _mesh_object_dict(key, None)
        if val is not None:
            return val
        
        self = super().__new__(cls)
        self.devices = devices.copy()
        self.devices.flags.writeable = False
        self.axis_names = axis_names
        self.axis_types = axis_types
        self._size = math.prod(self.shape.values()) if self.devices.ndim else 0

        # write it to the dict
        _mesh_object_dict[key] = self
        return self

    def __reduce__(self):
        return (_unpickle_mesh, (self.devices, self.axis_names, self.axis_types))

    def __eq__(self, other):
        # simple case
        if self is other:
            return True
        
        if not isinstance(other, JLEMesh):
            return False
        
        return (
            self.axis_names == other.axis_names and
            self.devices.shape == other.devices.shape and
            self.axis_types == other.axis_types and
            self._interal_device_list == other._internal_device_list
        )

    def __hash__(self):
        if not hasattr(self, '_hash'):
            self._hash = hash((self.axis_names, self._internal_device_list, self.devices.shape, self.axis_types))
        return self._hash
    
    def __setattr__(self, name, value):
        if hasattr(self, name):
            if getattr(self, name) == value:
                return
            raise RuntimeError(f"{name} mesh object already exists!")
        super().__setattr__(name, value)
    
    def __enter__(self):
        new_env = per_thread_resources.stack[-1].with_mesh(self)
        per_thread_resources.stack.append(new_env)
        per_thread_resources.env = new_env

        # need to update the config but will be done later
        return self

    def __exit__(self):
        per_thread_resources.stack.pop()
        per_thread_resources.env = per_thread_resources.stack[-1]

        # need to update the config but will be done later
        return False
    
    def update_data(self, 
                    devices=None, axis_names=None, axis_types=None):
        
        # one of the simplest function in the library
        if devices is None:
            devices = self.devices
        if axis_names is None:
            axis_names = self.axis_names
        if axis_types is None:
            axis_types = self.axis_types
        
        return JLEMesh(devices, axis_names, axis_types)

    # -------- basic methods from now on -------- #
    @property
    def size(self):
        return self._size
    
    @property
    def empty(self):
        return self.size == 0
    
    @property
    def axis_size(self):
        return self.devices.shape
    
    @property
    def shape(self):
        return collections.OrderedDict((name, size) for name, size in safe_zip(self.axis_names, self.devices.shape))
    
    @property
    def shape_in_tuple(self):
        return tuple((name, size) for name, size in safe_zip(self.axis_names, self.devices.shape))
    
    @property
    def local_mesh(self):
        return self._local_mesh(jlebc.process_index())

    def is_multi_process(self):
        return self.devices.size != len(self.local_devices)
    
    def _local_mesh(self, process_index):
        # TODO: need to write this function in this file
        return self._get_local_mesh(self, process_index)
    
    def device_ids(self):
        assert not self.empty
        return np.vectorize(lambda dev: dev.id, otypes=[int])(self.devices)
    
    def local_devices(self):
        return [dev for dev in self.devices.flat if dev.process_index == dev.client.process_index()]
    
    def _local_devices_in_set(self):
        return set(self.local_devices)
    
    def devices_flat_and_in_tuple(self):
        return tuple(self.devices.flat)
    
    def devices_flat_and_in_set(self):
        return set(self.devices.flat)
    
    def _internal_device_list(self):
        return xc.DeviceList(self.devices_flat_and_in_tuple)
    
    # -------- Repr and Str -------- #
    def __str__(self):
        if self.empty():
            return "JLEMesh()"
        mesh_str = ", ".join(f"'{k}': {v}" for k, v in self.shape.items())
        atr = f", axis_types={self.axis_types}"
        return f"JLEMesh({mesh_str}{atr})"
    
    def __repr__(self):
        if self.empty():
            return "JLEMesh(axis_names=(), axis_types=())"
        atr = f", axis_types={self.axis_types}"
        return (f"JLEMesh(axis_sizes={self.device_ids.shape}, axis_names={self.axis_names!r}{atr})")
    
# EMPTY ENV is needed for what?
EMPTY_ENV = JLEMeshEnvContext(JLEMesh(np.empty((), dtype=np.object_), ()))

class _PerThreadResources(threading.local):
    def __init__(self):
        self.stack = [EMPTY_ENV]
        self.env = self.stack[-1]

per_thread_resources = _PerThreadResources()