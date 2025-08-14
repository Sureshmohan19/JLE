"""miniJax.core.dtypes"""

from __future__ import annotations

import warnings
import builtins
import ml_dtypes
import functools
import numpy as np
from abc import abstractmethod
from typing import Any, Union, cast

from miniJax.core import config
from miniJax.core.typing import DTypeLike, DType
from miniJax.core.errors import TypePromotionError

x64_orNot: bool = config.enable_x64.value
int_info = ml_dtypes.iinfo
float_info = ml_dtypes.finfo

# Import custom FP8 and FP4 floating-point types from ml_dtypes
# and create local aliases for convenience.
float4_e2m1fn : type[np.generic] = ml_dtypes.float4_e2m1fn

float8_e3m4: type[np.generic] = ml_dtypes.float8_e3m4
float8_e4m3: type[np.generic] = ml_dtypes.float8_e4m3
float8_e5m2: type[np.generic] = ml_dtypes.float8_e5m2

float8_e4m3fn: type[np.generic] = ml_dtypes.float8_e4m3fn
float8_e8m0fnu: type[np.generic] = ml_dtypes.float8_e8m0fnu
float8_e4m3b11fnuz: type[np.generic] = ml_dtypes.float8_e4m3b11fnuz
float8_e4m3fnuz: type[np.generic] = ml_dtypes.float8_e4m3fnuz
float8_e5m2fnuz: type[np.generic] = ml_dtypes.float8_e5m2fnuz

bfloat16: type[np.generic] = ml_dtypes.bfloat16

int2: type[np.generic] = ml_dtypes.int2
int4: type[np.generic] = ml_dtypes.int4

uint2: type[np.generic] = ml_dtypes.uint2
uint4: type[np.generic] = ml_dtypes.uint4

# Instantiate and cache the NumPy dtype objects for each custom float type.
# While NumPy can often use the raw types, these explicit dtype objects are
# needed to inspect properties and ensure consistency.
_float4_e2m1fn_dtype: np.dtype = np.dtype(float4_e2m1fn)

_float8_e3m4_dtype: np.dtype = np.dtype(float8_e3m4)
_float8_e4m3_dtype: np.dtype = np.dtype(float8_e4m3)
_float8_e5m2_dtype: np.dtype = np.dtype(float8_e5m2)

_float8_e4m3fn_dtype: np.dtype = np.dtype(float8_e4m3fn)
_float8_e8m0fnu_dtype: np.dtype = np.dtype(float8_e8m0fnu)
_float8_e4m3b11fnuz_dtype: np.dtype = np.dtype(float8_e4m3b11fnuz)
_float8_e4m3fnuz_dtype: np.dtype = np.dtype(float8_e4m3fnuz)
_float8_e5m2fnuz_dtype: np.dtype = np.dtype(float8_e5m2fnuz)

_bfloat16_dtype: np.dtype = np.dtype(bfloat16)

_int2_dtype: np.dtype = np.dtype(int2)
_int4_dtype: np.dtype = np.dtype(int4)

_uint2_dtype: np.dtype = np.dtype(uint2)
_uint4_dtype: np.dtype = np.dtype(uint4)

# Group these scalar types and dtypes into respective groups
# We will need this a lot in the later sections
_custom_float_scalar_types = [
    float4_e2m1fn,
    float8_e3m4,
    float8_e4m3,
    float8_e5m2,
    float8_e4m3fn,
    float8_e8m0fnu,
    float8_e4m3b11fnuz,
    float8_e4m3fnuz,
    float8_e5m2fnuz,
    bfloat16,
]
_custom_float_dtypes = [
    _float4_e2m1fn_dtype,
    _float8_e3m4_dtype,
    _float8_e4m3_dtype,
    _float8_e5m2_dtype,
    _float8_e4m3fn_dtype,
    _float8_e8m0fnu_dtype,
    _float8_e4m3b11fnuz_dtype,
    _float8_e4m3fnuz_dtype,
    _float8_e5m2fnuz_dtype,
    _bfloat16_dtype,
]
_float4_dtypes = [_float4_e2m1fn_dtype]
_float8_dtypes = [
    _float8_e3m4_dtype,
    _float8_e4m3_dtype,
    _float8_e5m2_dtype,
    _float8_e4m3fn_dtype,
    _float8_e8m0fnu_dtype,
    _float8_e4m3b11fnuz_dtype,
    _float8_e4m3fnuz_dtype,
    _float8_e5m2fnuz_dtype
]
_custom_int_scalar_types = [ 
    int2, 
    int4,
]
_custom_uint_scalar_types = [
    uint2, 
    uint4
]
_int_dtypes = [
    _int2_dtype,
    _int4_dtype,
    _uint2_dtype,
    _uint4_dtype
]

# default types. 32bit is the default type in miniJax
# to prioritise performance and memory
bool_ = np.bool_
int_: type[Any] = np.int32
uint: type[Any] = np.uint32
float_: type[Any] = np.float32
complex_: type[Any] = np.complex64

_default_types: dict[str, type[Any]] = {
    'b': bool_,
    'i': int_,
    'u': uint,
    'f': float_,
    'c': complex_
}
_DEFAULT_TYPEMAP: dict[type, DTypeLike] = {
    bool: bool,
    int: int_,
    float: float_,
    complex: complex_
}
# dictionary to easily change from 64 to 32 
# and also to inexact
_change_64_to_32: dict[DType, DType] = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64')
}
_change_to_inexact: dict[DType, DType] = {
    np.dtype(k): np.dtype(v) for k,v in [
        ('bool', 'float32'),
        ('uint8', 'float32'), ('int8', 'float32'),
        ('uint16', 'float32'), ('int16', 'float32'),
        ('uint32', 'float32'), ('int32', 'float32'),
        ('uint64', 'float64'), ('int64', 'float64'),
    ]
}
# We use 64-bit types for int, float, and complex to match NumPy's defaults
# and ensure consistent precision across platforms.
py_scalar_types: dict[type, DType] = {
    bool: np.dtype('bool'),
    int: np.dtype('int64'),
    float: np.dtype('float64'),
    complex: np.dtype('complex128')
}
# incase if you are confused again and again seeing DType, its nothing other than np.dtype
float0: np.dtype = np.dtype([('float0', np.void, 0)])
miniJaxType = Union[type, DType] 
_weakTypes: list[miniJaxType] = [int, float, complex]
_registered_weakTypes: list[miniJaxType] = []
_boolTypes: list[miniJaxType] = [np.dtype(bool)]
_signedTypes: list[miniJaxType]
_unsignedTypes: list[miniJaxType]
_intTypes = list[miniJaxType]
_floatTypes: list[miniJaxType]
_complexTypes: list[miniJaxType]
_stringTypes: list[miniJaxType] = []

_unsignedTypes = [
    np.dtype(uint2),
    np.dtype(uint4),
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64')
]
_signedTypes = [
    np.dtype(int2),
    np.dtype(int4),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64')
]
_intTypes = _unsignedTypes + _signedTypes
_floatTypes = [
    *_custom_float_dtypes,
    np.dtype('float16'),
    np.dtype('float32'),
    np.dtype('float64')
]
_complexTypes = [
    np.dtype('complex64'),
    np.dtype('complex128')
]
if hasattr(np.dtypes, 'StringDType'):
    _stringTypes: list[miniJaxType] = [np.dtypes.StringDType()]

# miniJax dtype set includes float0(will do later) and _stringTypes
_miniJax_dtype_set = {
    float0,
    *_boolTypes,
    *_intTypes,
    *_floatTypes,
    *_complexTypes,
    *_stringTypes
}
# similar to _miniJax_dtypes_set, but no float0 and string types to the _miniJax_types,
# following the same principle as Jax to make sure the array can only be numeric
_miniJax_types = (
    _boolTypes +
    _intTypes +
    _floatTypes + 
    _complexTypes
)
_dtype_kinds: dict[str, set] = {
    'bool' : {*_boolTypes},
    'signed integer': {*_signedTypes},
    'unsigned integer': {*_unsignedTypes},
    'integral' : {*_unsignedTypes, *_signedTypes},
    'real floating': {*_floatTypes},
    'complex floating': {*_complexTypes},
    'numeric': {*_unsignedTypes, *_signedTypes, *_floatTypes, *_complexTypes}
}

class extended(np.generic):
    """
    will do nothing except useful for checking issubdtype.
    You can ask why not use numpy to check the subtype but there is
    a catch here - miniJax like Jax supports more ml_dtypes than normal
    Numpy do. So we have to make our own system to check.
    """

class ExtendedDType():
    """Abstract base class for extended dtypes"""
    @property
    @abstractmethod
    def type(self) -> type: ...

IS_SUBDTYPE_TYPES = (type, np.dtype, ExtendedDType)

# As we already mentioned above, 32bit is the default one in miniJax. 
# (TODO: need to understand how x64_orNot reacts when the env value 
# changes when loading the function. If it doesn't change dynamically, 
# then we really need to delete this variable and use config.enable_x64.value
def default_int_dtype() -> DType:
    return np.dtype(np.int64) if x64_orNot else np.dtype(np.int32)

def default_uint_dtype() -> DType:
    return np.dtype(np.uint64) if x64_orNot else np.dtype(np.uint32)

def default_float_dtype() -> DType:
    return np.dtype(np.float64) if x64_orNot else np.dtype(np.float32)

def default_complex_dtype() -> DType:
    return np.dtype(np.complex128) if x64_orNot else np.dtype(np.complex64)

# dtype conversion
def to_numeric(dtype: DTypeLike) -> DType:
    dtype_ = np.dtype(dtype)
    return np.dtype('int32') if dtype_ == np.dtype('bool') else dtype_

def to_inexact(dtype: DTypeLike) ->DType:
    dtype_ = np.dtype(dtype)
    return _change_to_inexact.get(dtype_, dtype_)

def to_floating(dtype: DTypeLike) -> DType:
    dtype_ = np.dtype(dtype)
    return float_info(_change_to_inexact.get(dtype_, dtype_)).dtype

def to_complex(dtype: DTypeLike) -> DType:
    flo = to_inexact(dtype)
    return np.dtype('complex128') if flo in [np.dtype('float64'), np.dtype('complex128')] else np.dtype('complex64')

# the following two are useful but most probably not needed.
def ctypes_supports_inf(dtype: DTypeLike) -> bool:
    typ = np.dtype(dtype).type
    if typ in {float4_e2m1fn, float8_e4m3fn, float8_e4m3b11fnuz, 
               float8_e4m3fnuz, float8_e5m2fnuz, float8_e8m0fnu}:
        return False
    return issubdtype(dtype, np.inexact)

def bit_width(dtype: DTypeLike) -> int:
    """
    we are getting bit width from ml_dtypes and not a custom implementaion. 
    Its actually much better approach because the custom types are defined there.
    """
    if dtype == np.dtype(bool):
        return 8
    elif issubdtype(dtype, np.integer):
        return int_info(dtype).bits
    elif issubdtype(dtype, np.floating):
        return float_info(dtype).bits
    elif issubdtype(dtype, np.complexfloating):
        return 2* float_info(dtype).bits
    else:
        raise ValueError(f"check the dtype please as '{dtype} is unrecognised'")

# main functions are starting here
def miniJax_dtype(type_obj: DTypeLike | None, align: bool = False, copy: bool = False) -> DType:
    if type_obj is None:
        type_obj = float_
    elif issubdtype(type_obj, extended):
        return type_obj
    elif isinstance(type_obj, type):
        type_obj = _DEFAULT_TYPEMAP.get(type_obj, type_obj)
    return np.dtype(type_obj, align=align, copy=copy)

# existing default functions in Numpy but modified for our use case 
# to accomodate custom types from ml_dtypes that we are using
def _issubclass(a: Any, b: Any) -> bool:
    """Same as default python issubclass. Infact, if you look at the code,
    you can see we are just using issubclass only. 
    The only caveat here is it wont return error and just return bool type
    """
    try:
        return issubclass(a,b)
    except TypeError:
        return False
    
def issubdtype(a: DTypeLike | ExtendedDType = float_,
               b: DTypeLike | ExtendedDType = float_
) -> bool:
    """
    Numpy allows None because they will convert None to float64 by default. 
    But here, we are removing None and making float_ as default. In case you are 
    wondering what is float_, its just float32 which is default in miniJax and I 
    think its the same in Jax as well
    """
    return _issubtype_cached(
        a if isinstance(a, IS_SUBDTYPE_TYPES) else np.dtype(a),
        b if isinstance(b, IS_SUBDTYPE_TYPES) else np.dtype(b)
    )

@functools.lru_cache(512)
def _issubtype_cached(
    a: type | np.dtype | ExtendedDType,
    b: type | np.dtype | ExtendedDType
) -> bool:
    """
    Theoretically we can place this function inside of issubdtype but for caching
    performance, we are separating it here. And honestly I think that's the only 
    reason I know of.
    """
    a_is_type = isinstance(a, type)
    b_is_type = isinstance(b, type)

    # just the same block as in Jax
    if b_is_type and _issubclass(b, extended):
        if isinstance (a, ExtendedDType):
            return _issubclass(a.type, b)
        if a_is_type and _issubclass(a, np.generic):
            return _issubclass(a, b)
        return _issubclass(np.dtype(a).type, b)
    if isinstance(b, ExtendedDType):
        return isinstance(a, ExtendedDType) and a==b
    if isinstance(a, ExtendedDType):
        a = a.type
        a_is_type = isinstance(a, type)

    a_scalar = a if a_is_type and _issubclass(a, np.generic) else np.dtype(a).type
    b_scalar = b if b_is_type and _issubclass(b, np.generic) else np.dtype(b).type

    if a_scalar in _custom_float_scalar_types:
        return b_scalar in {a_scalar, np.floating, np.inexact, np.number, np.generic}
    if a_scalar in _custom_int_scalar_types:
        return b_scalar in {a_scalar, np.signedinteger, np.integer, np.number, np.generic}
    if a_scalar in _custom_uint_scalar_types:
        return b_scalar in {a_scalar, np.unsignedinteger, np.integer, np.number, np.generic}
    
    return bool(np.issubdtype(a_scalar, b_scalar))
    
# canonicalize dtypes
# probably one of the most important API in this file
@functools.cache
def _canonicalize_dtypes(x64_enabled: bool, allow_extended: bool, dtype: Any) -> DType | ExtendedDType:
    """
    For me and other developers if anyone is there, the private function params and public api function
    params are reversed to make sure they are not confused. 
    Jax writers are so good man. Kudos!
    """
    if issubdtype(dtype, extended):
        if not allow_extended:
            raise ValueError(f"Canonicalize dtypes method called with allow_extended=False "
                             "but the dtype provided is indeed an extended dtype")
        return dtype
    
    try:
        dtype_ = np.dtype(dtype)
    except TypeError as e:
        raise TypeError(f"Numpy.dtype failed to recoginise your dtype") from e
    
    return dtype_ if x64_enabled else _change_64_to_32.get(dtype_, dtype_)

# public canonicalize_dtype function
def canonicalize_dtype(dtype: Any, allow_extended: bool = False) -> DType | ExtendedDType:
    return _canonicalize_dtypes(x64_orNot, allow_extended, dtype)

# dtype related functions
def isdtype(dtype: DTypeLike, kind: str | DTypeLike | tuple[str | DTypeLike, ...]) -> bool:
    options: set[DType] = set()
    dtype_ = np.dtype(dtype)
    kind_tuple: tuple[str | DTypeLike, ...] = ( kind if isinstance(kind, tuple) else (kind,))

    # walk through each kind
    for kind in kind_tuple:
        if isinstance(kind, str) and kind in _dtype_kinds:
            options.update(_dtype_kinds[kind])
            continue
        try:
            _dtype = np.dtype(kind)
        except TypeError as e:
            if isinstance(kind, str):
                raise ValueError(f"unrecognizedd kind:'{kind}' provided. please provide one in {list(_dtype_kinds.keys())}")
            raise TypeError(f"Expected kind to be str, dtype or tuple but instead got '{kind}'") from e
        options.add(_dtype)
    return dtype_ in options

def _miniJax_type(dtype: DType, weak_type: bool) -> miniJaxType:
    """
    Sometimes the return type if weak_type gets confused. 
    So a little explanation is below:

    return type(dtype.type(0).item())

    dtype.type gives the NumPy scalar class, e.g., np.int32.
    .type(0) creates a scalar of that dtype.
    .item() converts it to a Python scalar.
    type(...) gets the Python type (int or float).
    """
    # if weak type
    if weak_type:
        if dtype == bool:
            return dtype
        if dtype in _custom_float_dtypes:
            return float
        return type(dtype.type(0).item())
    # if not weak type
    return dtype

# main dtype function - just like numpy.dtype
def dtype(x: Any, canonicalize: bool = False) -> DType:
    if x is None:
        raise ValueError(f"Invalid type obj='{x}' provided to dtype.")
    
    is_type = isinstance(x, type)
    
    if is_type and x in py_scalar_types:
        dt = py_scalar_types[x]
    elif type(x) in py_scalar_types:
        dt = py_scalar_types[type(x)]
    elif is_type and _issubclass(x, np.generic):
        return np.dtype(x)
    elif issubdtype(getattr(x, 'dtype', None), extended):
        dt = x.dtype
    else:
        try:
            dt = np.result_type(x)
        except TypeError as e:
            raise TypeError(f"Cannot determine dtype of {x}") from e
        
    # after all this, now check whether its actually in jax types. 
    if dt not in _miniJax_dtype_set and not issubdtype(dt, extended):
        raise TypeError(f"Value:{x} with dtype: {dt} is not valid.")
    
    return canonicalize_dtype(dt, allow_extended=True) if canonicalize else dt

# scalar functions
def scalar_type(x: Any) -> type:
    typ_ = dtype(x)
    if typ_ in _custom_float_dtypes:
        return float
    elif typ_ in _int_dtypes:
        return int
    elif np.issubdtype(typ_, np.bool_):
        return bool
    elif np.issubdtype(typ_, np.integer):
        return int
    elif np.issubdtype(typ_, np.floating):
        return float
    elif np.issubdtype(typ_, np.complexfloating):
        return complex
    else:
        raise TypeError(f'Invalid value x={x} provided')
    
def _scalar_to_dtype(typ: type, value: Any = None) -> DType:
    dtype = canonicalize_dtype(py_scalar_types[typ])
    if type is int and value is not None:
        intinfo = np.iinfo(dtype)
        if value < intinfo.min or value > intinfo.max:
            raise OverflowError(f"Python int {value} is too large to convert to {dtype}")
        
    return dtype

def is_python_scalar(x: Any) -> bool:
    return type(x) in py_scalar_types

def check_valid_dtypes(dtype: DType) -> None:
    if dtype not in _miniJax_dtype_set:
        raise TypeError(f"dtype: '{dtype}' is not a valid miniJax type")
    
def is_string_dtype(dtype: DTypeLike) -> bool:
    return dtype in _stringTypes

def coerce_to_array(x: Any, dtype: DTypeLike | None = None) -> np.ndarray:
    if dtype is not None and type(x) in py_scalar_types:
        dtype = _scalar_to_dtype(type(x), x)
    return np.asarray(x, dtype)

# weak types
def register_weakType(typ: type) -> None:
    _registered_weakTypes.append(typ)

def is_weakly_typed(x: Any) -> bool:
    x_ = type(x)
    if x_ in _weakTypes or x_ in _registered_weakTypes:
        return True
    
def short_dtype_name(dtype) -> str:
    if isinstance(dtype, extended):
        return str(dtype)
    else:
        name = dtype.name
        return (
            name.replace('float', 'f')
                .replace('uint', 'u')
                .replace('int', 'i')
                .replace('complex', 'c')
        )
    
# type promotion related code and functions. 
# important ones so read carefully and understand/improve this
def _type_promotion_lattice(method: str) -> dict[miniJaxType, list[miniJaxType]]:
    # first destructure the _miniJax_types
    assert method in ('standard', 'strict'
    ), f"the entered method value={method} is not valid.'standard' or 'strict' are the two possible ones" 
    b1, = _boolTypes
    uint2, uint4, u1, u2, u4, u8, int2, int4, i1, i2, i4, i8 = _intTypes
    *custom_ftypes, bf, f2, f4, f8 = _floatTypes
    c4, c8 = _complexTypes
    i_, f_, c_ = _weakTypes

    out: dict[miniJaxType, list[miniJaxType]]
    if method == "standard":
        out = {
            b1: [i_],
            i_: [u1, uint2, uint4, i1, int2, int4],
            uint2: [], uint4: [], 
            u1: [i2, u2],
            u2: [i4, u4],
            u4: [i8, u8],
            u8: [f_],
            int2: [], int4: [],
            i1: [i2],
            i2: [i4],
            i4: [i8],
            i8: [f_],
            f_: [*custom_ftypes, bf, f2, c_],
            **{typ: [] for typ in custom_ftypes},
            bf: [f4],
            f2: [f4],
            f4: [f8, c4],
            f8: [c8],
            c_: [c4],
            c4: [c8],
            c8:[]
        }
        return out
    elif method == "strict":
        out = {
            i_: [f_] + _intTypes,
            f_: [c_] + _floatTypes,
            c_:_complexTypes,
            **{typ: [] for typ in _miniJax_types}
        }
        return out
    else:
        raise ValueError(f"invalid method:{method} provided")
    
def _make_lattice_upper_bounds(method: str) -> dict[miniJaxType, set[miniJaxType]]:
    lattice = _type_promotion_lattice(method)
    upper_bounds = {node: {node} for node in lattice}

    # walk through the lattice
    for lat in lattice:
        while True:
            new_upper_bounds = set()
            for b in upper_bounds[lat]:
                new_upper_bounds.update(lattice[b])
            if lat in new_upper_bounds:
                raise ValueError(f"cycle detected in DAG graph")
            if new_upper_bounds.issubset(upper_bounds[lat]):
                break
            upper_bounds[lat] |= new_upper_bounds

    return upper_bounds

_lattice_upper_bounds: dict[str, dict[miniJaxType, set[miniJaxType]]] = {
    "standard": _make_lattice_upper_bounds('standard'),
    "strict": _make_lattice_upper_bounds('strict')
}

def _least_upper_bound(method: str, *nodes: miniJaxType) -> miniJaxType:
    N = set(nodes)
    UB = _lattice_upper_bounds(method)

    try:
        bounds = [UB[n] for n in N]
    except KeyError:
        dtype_ = [n for n in N if n not in UB]
        raise ValueError(f"{dtype_} is not a valid dtype for miniJax type Promotion")
    
    CUB = set.intersection(*bounds)
    LUB = (CUB & N) or {c for c in CUB if CUB.issubset(UB[c])}
    if len(LUB) == 1:
        return LUB.pop()
    elif len(LUB) == 0:
        if config.numpy_dtype_promotion.value == "strict":
            e = (f"Input dtypes {tuple(str(n) for n in nodes)} have no available implicit dtype promotion "
                 "when method='strict'. Either try with 'standar' or use direct casting")
        elif any(n in _float8_dtypes for n in nodes):
            e = (f"Input dtypes {tuple(str(n) for n in nodes)} have no implicit dtype promotion path. "
                 "use direct casting if possible")
        elif any(n in _float4_dtypes for n in nodes):
            e = (f"Input dtypes {tuple(str(n) for n in nodes)} have no implicit dtype promotion path. "
                 "use direct casting if possible")
        elif any(n in _int_dtypes for n in nodes):
            e = (f"Input dtypes {tuple(str(n) for n in nodes)} have no implicit dtype promotion path. "
                 "use direct casting if possible")
        else:
            e = (f"Input dtypes {tuple(str(n) for n in nodes)} have no implicit dtype promotion path. "
                 "use direct casting if possible")
        raise TypePromotionError(e)
    else:
        raise TypePromotionError(f"what the hell! something went wrong internally. FYI: nodes]{nodes}")
    
def promote_types(a: DTypeLike, b: DTypeLike) -> DType:
    method: str = config.numpy_dtype_promotion.value
    a_ = cast(miniJaxType, a if any(a is t for t in _weakTypes) else np.dtype(a))
    b_ = cast(miniJaxType, b if any(b is t for t in _weakTypes) else np.dtype(b))
    return np.dtype(_least_upper_bound(method, a_, b_))

def _dtype_and_weaktype(value: Any) -> tuple[DType, bool]:
    return dtype(value), any(value is typ for typ in _weakTypes) or is_weakly_typed(value)

def _lattice_result_type(*args: Any) -> tuple[DType, bool]:
    method: str = config.numpy_dtype_promotion.value
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))

    # only one dtype
    if len(dtypes) == 1:
        out_dtype = dtypes[0]
        out_weak_type = weak_types[0]
    
    # one type dtype and atleast one strong type
    elif len(set(dtypes)) ==  1 and not all (weak_types):
        out_dtype = dtypes[0]
        out_weak_type = False
    
    # all weak type
    elif all(weak_types) and method != "strict":
        result = _least_upper_bound(method, *(_miniJax_type(dtype, False) for dtype in dtypes))
        out_dtype = dtype(result)
        out_weak_type = True
    
    else:
        result = _least_upper_bound(method, *(_miniJax_type(d, w) for d, w in zip(dtypes, weak_types)))
        out_dtype = dtype(result)
        out_weak_type = any(result is t for t in _weakTypes)
    return out_dtype, (out_dtype != bool_) and out_weak_type

def result_type(*args: Any, return_weak_type: bool = False) -> DType | tuple[DType, bool]:
    if len(args) == 0:
        raise ValueError(f"atleast one dtype or array is needed")
    dtype: DType | ExtendedDType
    dtype, weak_type = _lattice_result_type(*(float_ if arg is None else arg for arg in args))

    if weak_type:
        dtype = canonicalize_dtype(_default_types['f' if dtype in _custom_float_dtypes else dtype.kind])
    else:
        dtype = canonicalize_dtype(dtype, allow_extended=True)
    
    return (dtype, weak_type) if return_weak_type else dtype


# users might want these functions 
def check_user_dtype_supported(dtype):
    if issubdtype(dtype, extended):
        return
    if isinstance(dtype, type) and dtype in {bool, int, float, builtins.complex}:
        return
    np_type = np.dtype(dtype)
    is_custom_type = np_type.type in [*_custom_float_scalar_types, int2, int4, uint2, uint4]
    if (np_type.kind not in 'biufcT' and not is_custom_type and not dtype==float0):
        raise TypeError(f"Not valid dtype={dtype} provided")
    if dtype is not None and np_type != canonicalize_dtype(np_type):
        truncated_dtype = canonicalize_dtype(np_type).name
        warnings.warn(
            f"Requested dtype {dtype} will be truncated to {truncated_dtype}. "
            "Enable more dtypes with enable_x64 env"
        )

def can_cast_safely(a: Any, b: Any) -> bool:
    a_dtype = dtype(a, canonicalize=True)
    b_dtype = dtype(b, canonicalize=True)
    if a_dtype == b_dtype:
        return True
    return result_type(a, b_dtype) == b_dtype