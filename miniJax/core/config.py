"""miniJax.config"""

from __future__ import annotations

import os
import contextlib
import threading
import warnings
from typing import Any, TypeVar, Callable, Sequence, Protocol, NoReturn, cast, Optional

_T = TypeVar("_T")
_NO_DEFAULT_SENTINEL = object()

class miniJaxValueHolder(Protocol[_T]):
    """Protocol for configuration value holders."""
    value: _T

    def _set(self, value: _T) -> None: ...

class _miniJaxConfigBase:
    """Base class for all miniJax configuration implementations.

    This abstract class defines the common interface and shared functionality 
    for all configuration objects used within the miniJax framework.
    """
    def __init__(
            self,
            name: str,
            default: _T,
            validator: Callable[[Any], None] | None = None
    ):
        self._name = name
        self._global_value = default
        self._validator = validator

        if self._validator:
            self._validator(default)

    @property
    def value(self) -> _T:
        raise NotImplementedError("Subclasses of _miniJaxConfigBase class must implement 'Value' property")
    
    def _set(self, new_value: _T) -> None:
        raise NotImplementedError("Subclasses of _miniJaxConfigBase class must implement '_set' property")
    
    def set_global(self, value: _T) -> None:
        if self._validator:
            self._validator(value)
        self._global_value = value

    def get_global(self) -> _T:
        return self._global_value
    

class miniJaxConfig:
    """Central configuration object for miniJax.

    Holds and manages all individual configuration options, including both
    Flag and State instances. Provides a unified interface for accessing,
    modifying, and validating configuration values across the framework.
    """
    def __init__(self) -> None:
        self._value_holders: dict[str, miniJaxValueHolder] = {}
        self.meta: dict[str, tuple] = {}
        self._contextmanager_flags: set[str] = set()

    def add_option(
            self,
            name: str,
            holder: miniJaxValueHolder,
            opt_type: Any,
            meta_args: list = None,
            meta_kwargs: dict = None
    ) -> None:
        if name in self._value_holders:
            raise ValueError(f"Config name '{name} already defined'")

        self._value_holders[name] = holder
        self.meta[name] = (opt_type, meta_args or [], meta_kwargs or {})
        setattr(self, name, property(lambda self, h=holder: h.value))

    def update(self, name: str, value: Any) -> None:
        if name not in self._value_holders:
            raise AttributeError(f"Unrecognized config option: '{name}'")
        self._value_holders[name]._set(value)
    
    def read(self, name: str) -> Any:
        if name in self._contextmanager_flags:
            raise AttributeError(
                f"For flags with a corresponding contextmanager, read their value "
                f"via e.g. `config.{name}` rather than `config.FLAGS.{name}`."
            )
        return self._read(name)
    
    def _read(self, name: str) -> Any:
        try:
            return self._value_holders[name].value
        except KeyError:
            raise AttributeError(f"Unrecognised config option: '{name}'")
    
    @property
    def values(self) -> dict[str, Any]:
        return {name: holder.value for name, holder in self._value_holders.items()}


class State(_miniJaxConfigBase[_T]):
    """
    A configuration option whose value can be changed globally or
    temporarily overridden within a thread using a context manager.
    """
    def __init__(
            self, 
            name: str,
            default: _T,
            validator: Callable[[Any], None] | None = None,
            update_global_hook: Callable[[_T], None] | None = None,
            update_thread_local_hook:  Callable[[_T | None], None] | None = None,
            default_context_manager_value: Any = _NO_DEFAULT_SENTINEL,
    ):
        super().__init__(name, default, validator)

        self._thread_local = threading.local()
        self._update_global_hook = update_global_hook
        self._update_thread_local_hook = update_thread_local_hook
        self._default_context_manager_value = default_context_manager_value

        if self._update_global_hook:
            self._update_global_hook(default)

    @property
    def value(self) -> _T:
        if hasattr(self._thread_local, "value"):
            return self._thread_local.value
        return self._global_value
    
    def _set(self, new_value: _T) -> None:
        if self._validator:
            self._validator(new_value)
        self.set_global(new_value)
        if self._update_global_hook:
            self._update_global_hook(new_value)

    def set_global(self, value: _T) -> None:
        if self._validator:
            self._validator(value)
        self._global_value = value

    def set_local(self, value: Any) -> None:
        if value is _NO_DEFAULT_SENTINEL:
            if hasattr(self._thread_local, 'value'):
                delattr(self._thread_local, 'value')
        else:
            self._thread_local.value = value

    def swap_local(self, new_value: _T) -> Any:
        old_value = getattr(self._thread_local, 'value', _NO_DEFAULT_SENTINEL)
        self._thread_local.value = new_value
        return old_value
    
    def __bool__(self) -> NoReturn:
        raise TypeError(
            f"bool() not supported for instances of type '{type(self).__name__}' "
            f"(did you mean to use '{self._name}.value' instead?)"
        )
    
    def _add_hooks(self, update_global_hook, update_thread_local_hook):
        self._update_global_hook = update_global_hook
        self._update_thread_local_hook = update_thread_local_hook
        if update_global_hook:
            update_global_hook(self.get_global())

    def __call__(self, new_value: Any = _NO_DEFAULT_SENTINEL) -> StateContextManager:
        return StateContextManager(self, new_value)


class StateContextManager(contextlib.ContextDecorator):
    """Context manager for temporarily overriding State values."""

    def __init__(self, state: 'State', new_value: Any):
        self.state = state
        self.new_value = new_value
        self.old_value = None

        if new_value is _NO_DEFAULT_SENTINEL:
            if hasattr(state, '_default_context_manager_value') and state._default_context_manager_value is not _NO_DEFAULT_SENTINEL:
                self.new_value = state._default_context_manager_value
            else:
                raise TypeError(
                    f"Context manager for '{state._name}' config option requires "
                    "an argument representing the new value."
                )
        
        if state._validator:
            state._validator(self.new_value)

    def __enter__(self):
        self.old_value = self.state.swap_local(self.new_value)
        if hasattr(self.state, '_update_thread_local_hook') and self.state._update_thread_local_hook:
            self.state._update_thread_local_hook(self.new_value)
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.state.set_local(self.old_value)
        if hasattr(self.state, '_update_thread_local_hook') and self.state._update_thread_local_hook:
            if self.old_value is _NO_DEFAULT_SENTINEL:
                self.state._update_thread_local_hook(None)
            else:
                self.state._update_thread_local_hook(cast(Optional[Any], self.old_value))


config = miniJaxConfig()

# Environment Reading functions
def read_bool_env_value(varname: str, default: str) -> bool:
    val = os.getenv(varname, str(default))
    val = val.lower()

    if val in ('yes', 'true', 'on', '1'):
        return True
    elif val in ('no', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid bool value {val} for environment {varname}")

# STATE type - enum state
def enum_state(
        name: str,
        enum_values: Sequence[str],
        default: str,
        update_global_hook: Callable[[str], None] | None = None,
        update_thread_local_hook: Callable[[str | None], None] | None = None,
        extra_validator: Callable[[str], None] | None = None
) -> State[str]:
    """Create an enum-based configuration state."""
    if not isinstance(default, str):
        raise TypeError(f"Default value of enum_state must be of type str, got {type(default).__name__} instead")
    
    name = name.lower()
    default_from_env = os.getenv(name.upper(), default)

    if default_from_env not in enum_values:
        raise ValueError(f"Invalid value '{default_from_env} for config '{name}'. The possible values are '{list(enum_values)}''")
    
    config._contextmanager_flags.add(name)

    def validator(new_val):
        if type(new_val) is not str or new_val not in enum_values:
            raise ValueError(f"new enum values must be in {enum_values},"
                             f"but got {new_val} of type {type(new_val)}")

        if extra_validator is not None:
            extra_validator(new_val)

    s = State[str](
        name=name,
        default=default_from_env,
        validator=validator,
        update_global_hook=update_global_hook,
        update_thread_local_hook=update_thread_local_hook
    )

    config.add_option(
        name, s, 'enum', meta_args=[], meta_kwargs={"enum_values": list(enum_values)}
    )
    setattr(miniJaxConfig, name, property(lambda _: s.value))
    return s

# STATE type - bool state
def bool_state(
        name: str,
        default: bool,
        update_global_hook: Callable[[bool], None] | None = None,
        update_thread_local_hook: Callable[[bool | None], None] | None = None
) -> State[str]:
    """Create an bool-based configuration state."""
    if not isinstance(default, bool):
        raise TypeError(f"Default value in bool_state must be of bool type, not '{type(default)}")
    
    default = read_bool_env_value(name.upper(), default)
    name = name.lower()

    s = State[bool](
        name=name,
        default=default,
        update_global_hook=update_global_hook,
        update_thread_local_hook=update_thread_local_hook,
    )

    config.add_option(
        name, s, bool, meta_args=[], meta_kwargs={}
    )
    setattr(miniJaxConfig, name, property(lambda _: s.value))
    return s

# state and flag options
enable_x64 = bool_state(
    name='miniJax_enable_x64',
    default=False, # 32 bit is the standard in miniJax
)