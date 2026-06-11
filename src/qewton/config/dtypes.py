from typing import TYPE_CHECKING, Annotated, Any

# we trick the IDE to read type annotations as Annotated, while
# at runtime, we use seperate classes to make them hashable

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
