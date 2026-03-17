from pioneer.config.backend import DEFAULT_DL_BACKEND, TorchBackend
from pioneer.optim.trainer.trainable_parameters import _TrainableParameterBase


class Implementation:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "The __call__ method must be implemented by subclasses of Implementation."
        )


class TorchImplementation:
    """
    A PyTorch implementation consisting of a single torch.nn.Module
    """

    def __init__(self, torch_module) -> None:
        """Creates the underlying PyTorch module instance."""
        super().__init__()
        self.backend = TorchBackend
        self._torch_module = torch_module

    @property
    def torch_module(self):
        return self._torch_module

    def __call__(self, x):
        return self._torch_module(x)

    @property
    def trainable_parameters(self) -> _TrainableParameterBase:
        """Returns the trainable parameters of this node, which can be used for
        training the underlying algorithm (e.g. a neural network).

        Returns:
            _TrainableParameterBase: trainable parameters
        """
        return self._torch_module.parameters()
