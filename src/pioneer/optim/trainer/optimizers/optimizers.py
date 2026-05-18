from typing import Any

from ....config.backend import (
    Backend,
    TorchBackend,
    TensorflowBackend,
    DEFAULT_DL_BACKEND,
)


class Optimizer:

    existing_implementations = {}
    requires_closure = False

    def __init__(self, backend: Backend = DEFAULT_DL_BACKEND) -> None:
        self.backend = backend
        _ = self.backend.import_library()

    def build_optimizer(self):
        raise NotImplementedError("Optimizers are implemented via subclasses.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        optimizer_obj = self.build_optimizer()
        return optimizer_obj(*args, **kwds)


##################################################################################
# TODO: Add more optimizers and their respective implementations here.
# Also add some documentation.


# TODO: Maybe we need some ordering of the inputs again between different backends?
class Adam(Optimizer):

    def build_optimizer(self):
        if self.backend == TorchBackend:
            return self.backend.library.optim.Adam
        if self.backend == TensorflowBackend:
            return self.backend.library.keras.optimizers.Adam
        raise NotImplementedError(
            f"No implementation of Adam exists for backend {self.backend}."
        )


class SGD(Optimizer):

    def build_optimizer(self):
        if self.backend == TorchBackend:
            return self.backend.library.optim.SGD
        if self.backend == TensorflowBackend:
            return self.backend.library.keras.optimizers.SGD
        raise NotImplementedError(
            f"No implementation of SGD exists for backend {self.backend}."
        )


class LBFGS(Optimizer):

    def build_optimizer(self):
        if self.backend == TorchBackend:
            return self.backend.library.optim.LBFGS
        raise NotImplementedError(
            f"No implementation of LBFGS exists for backend {self.backend}."
        )

    requires_closure = True
