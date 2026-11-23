from typing import TYPE_CHECKING, Annotated, Any

# we trick the IDE to read type annotations as Annotated, while
# at runtime, we use separate classes to make them hashable

if TYPE_CHECKING:
    from typing import Annotated as BFloat16
    from typing import Annotated as Float16
    from typing import Annotated as Float32
    from typing import Annotated as Float64
    from typing import Annotated as Complex32
    from typing import Annotated as Complex64
    from typing import Annotated as Complex128
    from typing import Annotated as UInt8
    from typing import Annotated as UInt16
    from typing import Annotated as UInt32
    from typing import Annotated as UInt64
    from typing import Annotated as Int8
    from typing import Annotated as Int16
    from typing import Annotated as Int32
    from typing import Annotated as Int64
    from typing import Annotated as Number
    from typing import Annotated as Bool

else:

    class _DType:
        def __class_getitem__(cls, params):
            if not isinstance(params, tuple):
                params = (params,)
            # Erzeugt Annotated[BaseType, MarkerClass, Metadata...]
            # TODO: the classes are currently not being used in the dataconfig
            return Annotated[params[0] if params else Any, params[1], cls, params[2:]]

    class BFloat16(_DType):
        pass

    class Float16(_DType):
        pass

    class Float32(_DType):
        pass

    class Float64(_DType):
        pass

    class Complex32(_DType):
        pass

    class Complex64(_DType):
        pass

    class Complex128(_DType):
        pass

    class UInt8(_DType):
        pass

    class UInt16(_DType):
        pass

    class UInt32(_DType):
        pass

    class UInt64(_DType):
        pass

    class Int8(_DType):
        pass

    class Int16(_DType):
        pass

    class Int32(_DType):
        pass

    class Int64(_DType):
        pass

    class Number(_DType):
        pass

    class Bool(_DType):
        pass


def merge_dtypes(first: Any, second: Any) -> Any:
    """The dtype that data of both `first` and `second` has.

    From general to special: `Any` (or None), `TensorType`, the tensor
    class of a backend (any class that is no built-in Python type). A
    union matches where one of its members matches; several matching
    members give their union.

    Args:
        first: A dtype of a configuration.
        second: Another one.

    Returns:
        Any: The more special of the two, or the union of the matches.

    Raises:
        DataConfigMismatchError: If no member of `first` matches one of
            `second`.
    """
    from functools import reduce
    from operator import or_

    from qewton.config.errors import DataConfigMismatchError

    matches = []
    for mine in _members(first):
        for theirs in _members(second):
            match = _more_special(mine, theirs)
            if match is not _NO_MATCH and match not in matches:
                matches.append(match)
    if not matches:
        raise DataConfigMismatchError(f"The dtypes {first} and {second} do not match.")
    return matches[0] if len(matches) == 1 else reduce(or_, matches)


def merge_element_dtypes(first: Any, second: Any) -> Any:
    """The element dtype that data of both `first` and `second` has.

    From general to special: None (open), `Number`, a concrete marker such
    as `Float32`.

    Raises:
        DataConfigMismatchError: If both are concrete and differ.
    """
    from qewton.config.errors import DataConfigMismatchError

    for general, special in ((first, second), (second, first)):
        if general is None or general is Number and special is not None:
            return special
    if first is second:
        return first
    raise DataConfigMismatchError(f"The element dtypes {first} and {second} do not match.")


_NO_MATCH = object()


#: Built-in types a dtype may name; loading takes them from this list only.
_BUILTINS = {t.__name__: t for t in (int, float, bool, complex, str, bytes, tuple, list, dict)}


def saved_dtype(dtype: Any) -> Any:
    """The form a dtype is saved in, without classes to import: a marker
    (`"Any"`, `"TensorType"`, `"None"`), a built-in type or the tensor class
    of a backend by name, a class of qewton (e.g. `Float32`) as it is, and a
    union, `Literal` or generic type (e.g. `tuple[int, ...]`) as its origin
    and arguments. See :func:`loaded_dtype`."""
    import types
    import typing

    from qewton.backends import BACKEND_DICT
    from qewton.backends.base import TensorType

    if dtype is None or dtype is type(None):
        return "None"
    if dtype is Any:
        return "Any"
    if dtype is TensorType:
        return "TensorType"
    origin = typing.get_origin(dtype)
    if origin is typing.Union or isinstance(dtype, types.UnionType):
        return {"union": [saved_dtype(member) for member in typing.get_args(dtype)]}
    if origin is typing.Literal:
        return {"literal": list(typing.get_args(dtype))}
    if origin is not None:
        return {
            "generic": saved_dtype(origin),
            "args": [saved_dtype(arg) for arg in typing.get_args(dtype)],
        }
    if dtype is Ellipsis:
        return "..."
    if _BUILTINS.get(getattr(dtype, "__name__", None)) is dtype:
        return {"builtin": dtype.__name__}
    for name, backend in BACKEND_DICT.items():
        if backend.default_dtype is dtype:
            return {"backend": name}
    return dtype


def loaded_dtype(saved: Any) -> Any:
    """The dtype saved as `saved` (see :func:`saved_dtype`)."""
    from functools import reduce
    from operator import or_
    import typing

    from qewton.backends import BACKEND_DICT
    from qewton.backends.base import TensorType

    markers = {"None": None, "Any": Any, "TensorType": TensorType, "...": Ellipsis}
    if isinstance(saved, str):
        return markers[saved]
    if not isinstance(saved, dict):
        return saved
    if "union" in saved:
        members = [loaded_dtype(m) for m in saved["union"]]
        return reduce(or_, (type(None) if m is None else m for m in members))
    if "literal" in saved:
        return typing.Literal[tuple(saved["literal"])]
    if "generic" in saved:
        return loaded_dtype(saved["generic"])[tuple(loaded_dtype(a) for a in saved["args"])]
    if "builtin" in saved:
        return _BUILTINS[saved["builtin"]]
    return BACKEND_DICT[saved["backend"]].default_dtype


def _members(dtype: Any) -> tuple:
    import types
    import typing

    if isinstance(dtype, types.UnionType) or typing.get_origin(dtype) is typing.Union:
        return typing.get_args(dtype)
    return (dtype,)


def _more_special(first: Any, second: Any) -> Any:
    """The more special of two single dtypes, or `_NO_MATCH`."""
    from qewton.backends.base import TensorType

    for general, special in ((first, second), (second, first)):
        if general is None or general is Any:
            return Any if special is None else special
        if general is TensorType and _is_tensor_class(special):
            return special
    if first is second or first == second:
        return first
    if isinstance(first, type) and isinstance(second, type):
        if _is_subtype(first, second):
            return first
        if _is_subtype(second, first):
            return second
    return _NO_MATCH


#: Python's numeric tower: an int is accepted where a float is expected,
#: as type checkers do (PEP 484).
_NUMBERS = (bool, int, float, complex)


def _is_subtype(special: type, general: type) -> bool:
    if special in _NUMBERS and general in _NUMBERS:
        return _NUMBERS.index(special) <= _NUMBERS.index(general)
    return issubclass(special, general)


def _is_tensor_class(dtype: Any) -> bool:
    return isinstance(dtype, type) and dtype.__module__ != "builtins"


def element_dtype_in(metadata) -> Any:
    """The element dtype marker (e.g. `Float32`) among the metadata of a type
    annotation such as ``Float32[TensorType, ...]``; None if there is none."""
    return next(
        (m for m in metadata if isinstance(m, type) and issubclass(m, _DType)), None
    )
