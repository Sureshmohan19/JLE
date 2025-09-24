"""JLE.core.ainfo"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any
import operator

import numpy as np

from JLE.core import dtypes
from JLE.core.memory import MemorySpace
from JLE.core.typing import Shape, DimSize
from jaxlib import utils as jaxlib_utils

AxisName = Hashable

safe_map, unsafe_map = jaxlib_utils.safe_map, map

def is_symbolic_dim(value: Any) -> bool:
    return hasattr(value, "dimension_as_value")

def is_constant_dim(value: DimSize) -> bool:
    try:
        operator.index(value)
        return True
    except:
        return False

def is_dimension(value: Any) -> bool:
    return is_symbolic_dim(value) or is_constant_dim(value)

def _dtype_object(dtype):
    assert dtype is not None, "dtype cannot be None"
    return dtype if isinstance(dtype, dtypes.ExtendedDType) else np.dtype(dtype)

def _standardize_dimension(dim: DimSize) -> DimSize:
    """Standardize the user provided dimension after checking them"""
    try:
        return operator.index(dim)
    except TypeError as err:
        type_error = err
    
    if is_dimension(dim):
        return dim
    else:
        raise type_error

def standardize_shape(shape: Shape, context: str="") -> tuple[Any, ...]:
    """Standardize the user provided shape after checking them"""
    if isinstance(shape, int):
        shape = (shape,)
    try:
        return tuple(unsafe_map(_standardize_dimension, shape))
    except TypeError:
        pass

    raise _raise_invalid_shape(shape, context)

def _raise_invalid_shape(shape: Shape, context: str = ""):
    """Raise invalid shape error if needed"""
    err = f"Shapes must be 1D sequence of integer values, not {shape}."
    if context:
        err += f" {context}. "

    return TypeError(err)

class AInfo:
    """Array Info Class which holds only some metadata about an array rather than the array itself.
    
    Exactly works like jax.aval
    """
    def to_tangent_ainfo(self):
        raise NotImplementedError("Subclasses must override this method")
    
    def update(self, **kwargs):
        raise NotImplementedError("Subclasses must override this method")
    
    def __repr__(self):
        try:
            items = getattr(self, '__dict__', {}).items()
            if items:
                args = ','.join(f'{k}={v!r}' for k,v in items)
                return f'{self.__class__.__name__}({args})'
        except Exception:
            pass
        return self.__class__.__name__
    
    def update_weak_type(self, weak_type):
        return self
    
    def strip_weak_type(self) -> AInfo:
        return self.update_weak_type(False)
    
    def normalize(self) -> AInfo:
        return self.strip_weak_type()
    
    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        return str(self)

class UnshapedArray(AInfo):
    __slots__ = ["dtype", "weak_type"]

    def __init__(self, dtype, weak_type=False):
        assert isinstance(weak_type, bool), "weak_type must be a boolean"
        self.dtype = _dtype_object(dtype)
        self.weak_type = weak_type
        raise Exception("UnshapedArray object should never be created")
    
    def update_weak_type(self, weak_type):
        return self.update(weak_type=weak_type)
    
    def to_tangent_ainfo(self):
        raise NotImplementedError("UnshapedArray cannot have tangent")

    def update(self, **kwargs):
        raise Exception("Cannot update UnshapedArray")
    
    def __eq__(self, other):
        return (
            type(self) is type(other) and 
            self.dtype == other.dtype and
            self.weak_type == other.weak_type
        )
    
    def __ne__(self, other):
        return not self == other
    
    def __hash__(self):
        return hash((self.dtype, self.weak_type))
    
    def __str__(self):
        prefix = "~" if self.weak_type else ""
        return f"{prefix}{self.dtype.name}"
    
    def str_short(self, short_dtypes=False, mesh_axis_types=False) -> str:
        dtype_str = dtypes.short_dtype_name(self.dtype) if short_dtypes else self.dtype.name
        prefix = "~" if self.weak_type else ""
        return f"{prefix}{dtype_str}"
    
# ------------ For ShapedArray ------------- #

def get_memory_space(space):
    assert isinstance(space, MemorySpace)
    return space

def get_sharding(sharding, shape):
    """Find the sharding and check/modifies it"""
    ndim = len(shape)
    

class ShapedArray(AInfo):
    __slots__ = ["dtype", "weak_type", "shape", "sharding", "vma", "memory_space"]

    def __init__(
            self, 
            shape, 
            dtype, 
            weak_type=False,
            sharding=None,     
            vma: frozenset[AxisName] = frozenset(),
            memory_space: MemorySpace = MemorySpace.Device,
        ):
        assert isinstance(weak_type, bool), "weak_type must be a boolean"
        self.shape = standardize_shape(shape)
        self.dtype = _dtype_object(dtype)
        self.weak_type = weak_type
        self.memory_space = get_memory_space(memory_space)
        #self.sharding = get_sharding(sharding, self.shape)
        #self.vma = get_vma(vma, self.sharding.mesh)
