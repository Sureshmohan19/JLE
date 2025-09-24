"""JLE.core.utils"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, cast

T = TypeVar("T")

def _unwrap_func(obj: Callable) -> Callable:
    """Get the underlying function from property/cached_property if needed."""
    if isinstance(obj, property):
        return cast(property, obj).fget
    elif isinstance(obj, functools.cached_property):
        return obj.func
    return obj

def use_cpp_class(cpp_cls: type[Any]) -> Callable[[type[T]], type[T]]:
    """ A decorator which replaces Python class like JLEArray withits C++ implementation at runtime
    
    Regarding input and output types:
    takes a C++ class type (cpp_cls) and returns a decorator, 
    which itself takes a Python class (type[T]) and returns that same class type (type[T]).
    """
    def wrapper(cls: type[T]) -> type[T]:
        if cpp_cls is None:
            return cls
        
        exclude = {'__module__', '__dict__', '__doc__'}
        for name, attr in cls.__dict__.items():
            if name in exclude:
                continue
            if not hasattr(_unwrap_func(attr), "_use_cpp"):
                setattr(cpp_cls, name, attr)
        
        cpp_cls.__doc__ = cls.__doc__
        return cpp_cls
    
    return wrapper