import numpy as np
import pioneer


data = np.random.random((10, 3, 3, 6))

X = pioneer.configurations.Variable("x", 2)
U = pioneer.configurations.Variable("u", 1)
T = pioneer.configurations.Variable("t", 3)
dataset = pioneer.nodes.DataSet.from_data(data, X * U * T, batch_size=100)

slice_node = pioneer.nodes.SliceNode(dataset.data_config)

data_out = dataset()


# class CustomAlgo(pioneer.AlgorithmNode):
#     def run(self, inputs=None) -> dict[str, np.ndarray]:
#         if inputs is None:
#             return {}
#         x = inputs[self.InputKeys.INPUT]
#         return {self.OutputKeys.OUTPUT: x**2}

#     def setup(self) -> None:
#         return


# algo = CustomAlgo(X, X)
# control_node2 = pioneer.pipeline.ControlNode(
#     algo["output"].data_configuration, "SaveNode2"
# )
# pipeline = pioneer.pipeline.Pipeline()

# pipeline.connect(
#     dataset[dataset.OutputKeys.OUTPUT], control_node[control_node.InputKeys.INPUT]
# )
# pipeline.connect(control_node[control_node.OutputKeys.OUTPUT], algo["input"])
# pipeline.connect(algo["output"], control_node2["input"])

# pipeline.validate()

# runtime = pipeline.create_runtime()
# runtime.run()

# print(control_node.stored_data, control_node2.stored_data)
# train_data, vali_data, test_data = dataset()
