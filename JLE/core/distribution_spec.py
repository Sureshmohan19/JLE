"""JLE.core.distribution_spec"""

from __future__ import annotations

import jaxlib._jax as _jax

from JLE.core.utils import use_cpp_class, use_cpp_method

_UNCONSTRAINED_PARTITION = _jax.UNCONSTRAINED_PARTITION
_canonicalize_partition = _jax.canonicalize_partition

@use_cpp_class(_jax.PartitionSpec)
class DistributionSpec:
    """
    Tuple describing how to partition an array across a mesh of devices 
    - same words as JAX's PartitionSpec class and aimed to do similar here
    """

    @use_cpp_method()
    def __init__(self, *partitions, unreduced=frozenset(), reduced=frozenset()):
        # canonicalize the partitions
        # Validate that unreduced and reduced arguments are sets, then store as frozensets
        self._partitions = tuple(_canonicalize_partition(part) for part in partitions)
        assert isinstance(unreduced, (set, frozenset)), f"`unreduced` argument of DistributionSpec should be of type `frozenset` or `set`. Got type {type(unreduced)}"
        assert isinstance(reduced, (set, frozenset)), f"`reduced` argument of DistributionSpec should be of type `frozenset` or `set`. Got type {type(reduced)}"
        self.unreduced = frozenset(unreduced)
        self.reduced = frozenset(reduced)

    @use_cpp_method
    def __eq__(self, other):
        if isinstance(other, DistributionSpec):
            return (self._partitions == other._partitions and 
                    self.unreduced == other.unreduced and 
                    self.reduced == other.reduced)
        elif isinstance(other, tuple):
            assert not self.unreduced, f"other {other} cannot be of instance `tuple` when self {self} has unreduced in `__eq__` of DistributionSpec."
            assert not self.reduced, f"other {other} cannot be of instance `tuple` when self {self} has reduced in `__eq__` of DistributionSpec."
            other_cls = tuple(_canonicalize_partition(oth) for oth in other)
            return self._partitions == other_cls
        else:
            return False
    
    def __add__(self, other):
        if isinstance(other, DistributionSpec):
            return DistributionSpec(
                *self, *other,
                unreduced={*self.unreduced, *other.unreduced},
                reduced={*self.reduced, *other.reduced})
        elif isinstance(other, tuple):
            assert not self.unreduced, f"other {other} cannot be of instance `tuple` when self {self} has unreduced in `__add__` of DistributionSpec."
            assert not self.reduced, f"other {other} cannot be of instance `tuple` when self {self} has reduced in `__add__` of DistributionSpec."
            return DistributionSpec(*self, *other)
        else:
            raise NotImplementedError(f"Addition not supported between DistributionSpec and {type(other).__name__}. Only DistributionSpec and tuple are supported.")
        
    def __radd__(self, other):
        if not isinstance(other, tuple):
            raise NotImplementedError(f"Right addition not supported between {type(other).__name__} and DistributionSpec. Only tuple is supported.")
        
        assert not self.unreduced, f"other {other} cannot be of instance `tuple` when self {self} has unreduced in `__radd__` of DistributionSpec."
        assert not self.reduced, f"other {other} cannot be of instance `tuple` when self {self} has reduced in `__radd__` of DistributionSpec."
        return DistributionSpec(*other, *self)
    
    @use_cpp_method
    def __hash__(self):
        return hash((self._partitions, self.unreduced, self.reduced))
    
    def index(self, value):
        return self._partitions.index(_canonicalize_partition(value))
    
    def count(self, value):
        return self._partitions.count(_canonicalize_partition(value))
    
    def update(self, **kwargs):
        partitions = kwargs.pop("partitions", self._partitions)
        unreduced = kwargs.pop("unreduced", self.unreduced)
        reduced = kwargs.pop("reduced", self.reduced)
        return DistributionSpec(*partitions, unreduced=unreduced, reduced=reduced)
    
    def __repr__(self):
        pr = repr(self._partitions)[1:-1] # Remove outer parentheses from tuple repr
        if not self.unreduced and not self.reduced:
            return f"DistributionSpec({pr})"
        
        # Build unreduced/reduced string inline
        if self.unreduced and self.reduced:
            ur_str = f"unreduced={set(self.unreduced)!r}, reduced={set(self.reduced)!r}"
        elif self.unreduced and not self.reduced:
            ur_str = f"unreduced={set(self.unreduced)!r}"
        elif not self.unreduced and self.reduced:
            ur_str = f"reduced={set(self.reduced)!r}"
    
        pr = '' if not pr else f"{pr} " if pr.endswith(',') else f"{pr}, "
        return f"DistributionSpec({pr}{ur_str})"
    
    def __reduce__(self):
        return (DistributionSpec, (self._partitions, self.unreduced, self.reduced))
    
    # -------- Same old boring but useful methods -------- #
    def __getitem__(self, i):
        return self._partitions[i]
    
    def __iter__(self):
        return iter(self._partitions)

    def __len__(self):
        return len(self._partitions)
    
    def normalized_spec_for_aval(self, ndim: int) -> DistributionSpec:
        # Replace unconstrained partitions with None
        partitions = [None if p is _UNCONSTRAINED_PARTITION else p for p in self._partitions]
        
        # Pad with None if we have fewer partitions than dimensions
        if len(partitions) < ndim:
            partitions.extend([None] * (ndim - len(partitions)))

        return self.update(partitions=partitions)
    
DistributionSpec.__module__ = 'JLE.core.sharding'