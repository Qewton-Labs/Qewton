import numpy as np
import tensorflow as tf
import pioneer

x_data = np.linspace(0, 1, 1000).reshape(-1, 1)
u_data = x_data**2 + np.sin(6.0 * x_data)
data = tf.convert_to_tensor(np.column_stack((x_data, u_data)), dtype=tf.float32)

X = pioneer.config.Variable("x", 1)
U = pioneer.config.Variable("u", 1)
dataset = pioneer.nodes.DataSet.from_data(data, X * U, batch_size=1000)

slice_node = pioneer.nodes.SplitNode(dataset.data_config)

model = pioneer.algorithms.TFFCN(X, U, hidden_layers=2, hidden_neurons=8)

constrain = pioneer.constraints.MSEConstraint(
    model[model.OutputKeys.OUTPUT].data_configuration,
)

pipeline = pioneer.pipelines.Pipeline()

pipeline.connect(dataset[dataset.OutputKeys.OUTPUT], slice_node[dataset.InputKeys.INPUT])
pipeline.connect(slice_node["x"], model[dataset.InputKeys.INPUT])
pipeline.connect(slice_node["u"], constrain[constrain.InputKeys.INPUT1])
pipeline.connect(model[model.OutputKeys.OUTPUT], constrain[constrain.InputKeys.INPUT2])

pipeline.validate()
# runtime = pipeline.create_runtime()
# runtime.run()

trainer = pioneer.optim.trainer.TensorFlowTrainer(
    [pipeline],
    training_constraints=[constrain],
    optimizer_cls=tf.keras.optimizers.Adam,  # type: ignore
    max_iterations=5000,
    learning_rate=0.001,
    device="/CPU:0",
)

trainer.run()
