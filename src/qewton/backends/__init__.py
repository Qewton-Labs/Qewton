from .base import Backend, TensorType

# internal hierarchy: if torch exists, we set it as the default backend,
# otherwise if tensorflow exists, we set it as the default backend.
DEFAULT_DL_BACKEND = None

try:
    from .torch.base import TorchBackend

    DEFAULT_DL_BACKEND = TorchBackend
except (ImportError, AttributeError):
    pass
try:
    from .tensorflow.base import TensorflowBackend

    if DEFAULT_DL_BACKEND is None:
        DEFAULT_DL_BACKEND = TensorflowBackend
except (ImportError, AttributeError):
    pass
