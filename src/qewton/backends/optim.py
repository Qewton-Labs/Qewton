from qewton.backends.base import Backend, TensorType


class OptimBackend(Backend[TensorType]):
    @staticmethod
    def setup_optimizer():
        raise NotImplementedError

    @staticmethod
    def do_optimization_step():
        raise NotImplementedError

    @staticmethod
    def _cleanup():
        raise NotImplementedError

    @staticmethod
    def _closure():
        raise NotImplementedError

    # optim creators
    @staticmethod
    def adam():
        raise NotImplementedError

    @staticmethod
    def sgd():
        raise NotImplementedError

    @staticmethod
    def lbfgs():
        raise NotImplementedError
