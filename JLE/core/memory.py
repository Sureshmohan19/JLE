"""JLE.core.memory"""

import enum

class MemorySpace(enum.Enum):
    Device = enum.auto()
    Host = enum.auto()
    Any = enum.auto()

    def __repr__(self):
        return f"MemorySpace.{self.name}"