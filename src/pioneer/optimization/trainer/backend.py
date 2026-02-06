import torch

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

    @property
    def no_grad(self):
        raise NotImplementedError

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return []


class TorchBackend(BackendOptimizer):
    def __init__(self, optimizer_cls=torch.optim.Adam, lr=1e-3):
        self.optimizer_cls = optimizer_cls
        self.lr = lr
        self.optimizer: torch.optim.Optimizer

    def setup(self, parameters):
        self.optimizer = self.optimizer_cls(parameters, lr=self.lr)

    def compute_gradients(self, loss):
        loss.backward()

    def apply_gradients(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    @property
    def no_grad(self):
        return torch.no_grad()
