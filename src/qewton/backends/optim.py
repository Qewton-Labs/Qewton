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
    def adam(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def sgd(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def lbfgs(*args, **kwargs):
        raise NotImplementedError

    # lr scheduler creators
    @staticmethod
    def step_lr(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def exponential_lr(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def cosine_annealing_lr(*args, **kwargs):
        raise NotImplementedError
