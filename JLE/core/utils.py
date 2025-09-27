"""JLE.core.utils"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, cast

from jaxlib import utils as jaxlib_utils

T = TypeVar("T")

safe_zip = jaxlib_utils.safe_zip

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

def use_cpp_method(is_enabled: bool = True) -> Callable[[T], T]:
    assert isinstance(is_enabled, bool), "is_enabled argument in use_cpp_method should be a bool type"
    
    def decorator(meth):
        if is_enabled:
            unwrapped_function = _unwrap_func(meth)
            unwrapped_function._use_cpp = True
        return meth
    return decorator