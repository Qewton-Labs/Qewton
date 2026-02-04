from ..hyperparameter.base import HyperParameter


# TODO: Dont think this is an optimal structure for any optimizer...
class BackendOptimizer:

    def setup(self, parameters):
        """Register parameters in the backend."""
        raise NotImplementedError

    def compute_gradients(self, loss):
        raise NotImplementedError

    def apply_gradients(self):
        raise NotImplementedError

    def no_grad(self):
        raise NotImplementedError

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return []
