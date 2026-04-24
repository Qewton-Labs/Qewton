from typing import Any

from ....algorithms.backend import (
    Implementation,
    TorchImplementation,
    TensorflowImplementation,
    DEFAULT_DL_BACKEND,
)


class Optimizer:

    existing_implementations = {}
    requires_closure = False

    def __init__(self, backend: Implementation = DEFAULT_DL_BACKEND) -> None:
        self.backend = backend

    def build_optimizer(self):
        if self.backend in self.existing_implementations:
            optimizer_cls_name = self.existing_implementations[self.backend]
            return self.backend(optimizer_cls_name)
        raise NotImplementedError(
            f"No implementation of {self.__class__.__name__} exists for \
                backend {self.backend}."
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        optimizer_obj = self.build_optimizer()
        return optimizer_obj(*args, **kwds)


##################################################################################
# TODO: Add more optimizers and their respective implementations here.
# Also add some documentation.


# TODO: Maybe we need some ordering of the inputs again between different backends?
class Adam(Optimizer):

    existing_implementations = {
        TorchImplementation: "optim.Adam",
        TensorflowImplementation: "keras.optimizers.Adam",
    }


class SGD(Optimizer):

    existing_implementations = {
        TorchImplementation: "optim.SGD",
        TensorflowImplementation: "keras.optimizers.SGD",
    }


class LBFGS(Optimizer):

    existing_implementations = {
        TorchImplementation: "optim.LBFGS",
        # TensorFlow has no direct equivalent
    }
    requires_closure = True
