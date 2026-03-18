import tensorflow as tf

from ..config.variables import Variable
from ..optim.hyperparameter.base import (
    HyperParameter,
    DiscreteHyperparameter,
    CategoricalHyperparameter,
)


class TFFCN(AlgorithmNode):
    def __init__(
        self,
        input_variable: Variable,
        output_variable: Variable,
        hidden_layers: int | DiscreteHyperparameter,
        hidden_neurons: int | DiscreteHyperparameter,
        activation_fn: str | CategoricalHyperparameter = "tanh",
        name: str = "TFFCNNode",
    ) -> None:
        super().__init__(
            input_variable=input_variable, output_variable=output_variable, name=name
        )
        self.model: tf.keras.Model = None
        self.hidden_layer = HyperParameter.from_value(hidden_layers, "Hidden Layers")
        self.hidden_neurons = HyperParameter.from_value(hidden_neurons, "Hidden Neurons")
        self.activation_fn = HyperParameter.from_value(
            activation_fn, "Activation Function"
        )

    def setup(self) -> None:
        """Creates the underlying Keras model"""
        layers = []

        if self.hidden_layer.value > 0:
            # First hidden layer
            layers.append(
                tf.keras.layers.Dense(
                    self.hidden_neurons.value,
                    activation=self.activation_fn.value,
                    input_shape=(self.input_variable.dim,),
                    dtype=tf.float32,
                )
            )
            # Additional hidden layers
            for _ in range(self.hidden_layer.value - 1):
                layers.append(
                    tf.keras.layers.Dense(
                        self.hidden_neurons.value,
                        activation=self.activation_fn.value,
                        dtype=tf.float32,
                    )
                )
            # Output layer
            layers.append(
                tf.keras.layers.Dense(self.output_variable.dim, dtype=tf.float32)
            )
        else:
            # No hidden layer
            layers.append(
                tf.keras.layers.Dense(
                    self.output_variable.dim,
                    input_shape=(self.input_variable.dim,),
                    dtype=tf.float32,
                )
            )

        self.model = tf.keras.Sequential(layers)
        self._state = AlgorithmState.READY

    def _run(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        if self.state == AlgorithmState.UNINITIALIZED:
            self.setup()
        data = inputs[self.InputKeys.INPUT]
        outdata = self.model(data)
        return {self.OutputKeys.OUTPUT: outdata}

    @property
    def trainable_parameters(self):
        return self.model.trainable_variables
