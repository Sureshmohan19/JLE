"""JLE.core.typing"""

from collections.abc import Sequence
from typing import Any, Union, Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class SupportsDType(Protocol):
    """
    Any object with a .dtype property (like arrays) automatically 
    satisfies this protocol without explicit inheritance.

    For people who are new to runtime_checkable decorator, I can simply 
    say that, '@runtime_checkable means you can use isinstance() to check it at runtime'
    """
    @property
    def dtype(self) -> np.dtype: ...

DType = np.dtype
DTypeLike = Union[
    str, 
    type[Any], 
    np.dtype, 
    SupportsDType
]

# Shapes can simply be ints but sometimes we need to make sure 
# they support texts inside of that sequence
DimSize = Union[int, Any]
Shape = Sequence[DimSize]