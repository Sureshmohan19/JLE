"""JLE.core.basearray"""

from __future__ import annotations

from typing import Union
import numpy as np

from JLE.core.jaxlib import xla_client
from JLE.core.utils import use_cpp_class

class JLEArray:
    """Array Base Class
    
    Just like jax.Array instance check, the main use of this class is the following
    
    isinstance(x, jle.JLEArray)
    or
    x:JLEArray
    """

    @property
    def dtype(self) -> np.dtype:
        raise NotImplementedError('Subclass must implement dtype property')
    
    @property
    def ndim(self) -> int:
        raise NotImplementedError('Subclass must implement ndim property')
    
    @property
    def shape(self) -> tuple[int, ...]:
        """For future references, the return type
        tuple[int, ...] means 'a tuple of ints of arbitrary length'"""
        raise NotImplementedError('Subclass must implement shape property')
    
    @property
    def size(self) -> int:
        raise NotImplementedError('Subclass must implement size property')

JLEArray = use_cpp_class(xla_client.Array)(JLEArray)
JLEArray.__module__ = "jle"

StaticScalar = Union[
    np.bool_, np.number, # Numpy scalar types
    bool, int, float, complex # Native Python scalar types
    ]

JLEArrayLike = Union[
    JLEArray, # Our own array type
    np.ndarray, # Numpy array type
    StaticScalar # Valid scalar type
]