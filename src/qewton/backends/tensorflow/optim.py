import tensorflow as tf

from qewton.backends.optim import OptimBackend


class TensorFlowOptimBackend(OptimBackend[tf.Tensor]):
    adam = tf.keras.optimizers.Adam
    sgd = tf.keras.optimizers.SGD
