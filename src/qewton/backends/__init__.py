from .base import Backend, TensorType, DeepLearningBackend, ComputingBackend
from .numpy.base import NumPyBackend

# internal hierarchy: if torch exists, we set it as the default backend,
# otherwise if tensorflow exists, we set it as the default backend.
_backend_found = False

try:
    from .torch.base import TorchBackend

    DEFAULT_DL_BACKEND = TorchBackend
    _backend_found = True
except (ImportError, AttributeError):
    # Torch is not installed or failed to initialize (e.g. circular imports)
    pass

if not _backend_found:
    try:
        from .tensorflow.base import TensorflowBackend

        DEFAULT_DL_BACKEND = TensorflowBackend
        _backend_found = True
    except (ImportError, AttributeError):
        pass

if not _backend_found:
    DEFAULT_DL_BACKEND = DeepLearningBackend
