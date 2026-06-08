import tensorflow as tf
from qewton.backendss.base import Backend


class TensorflowBackend(Backend[tf.Tensor]):
    @classmethod
    def import_library(cls):
        if cls.library is None:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel # type: ignore

            cls.library = tf
        return cls.library

    @classmethod
    def standard_datatype(cls):
        if not cls.exists():
            raise ImportError("TensorFlow is not installed.")

        return cls.library.Tensor

    @classmethod
    def to(cls, data, device):
        return data

    @classmethod
    def from_numpy(cls, data):
        return cls.library.convert_to_tensor(data)


def get_dtype_tf():
    try:
        import tensorflow as tf

        return tf.Tensor
    except ImportError:
        return Any
