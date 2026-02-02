from typing import Any

import numpy as np
import pioneer


data = np.random.random((10, 3, 3, 2))

X = pioneer.data.Variable("x", 2)
dataset = pioneer.data.DataSet.from_data(data, X, batch_size=100)

# print(dataset.ports)

control_node = pioneer.pipeline.ControlNode(dataset.data_config, "SaveNode1")


class CustomAlgo(pioneer.AlgorithmNode):
    def run(self, inputs=None) -> dict[str, Any]:
        if inputs is None:
            return {}
        x = inputs["input"]
        return {"output": x**2}

    def setup(self) -> None:
        return


algo = CustomAlgo(X, X)
control_node2 = pioneer.pipeline.ControlNode(
    algo.output_ports["output"].data_configuration, "SaveNode2"
)
pipeline = pioneer.pipeline.Pipeline()

pipeline.connect(dataset.output_ports["test_data"], control_node.input_ports["input"])
pipeline.connect(control_node.output_ports["output"], algo.input_ports["input"])
pipeline.connect(algo.output_ports["output"], control_node2.input_ports["input"])
pipeline.validate()
runtime = pipeline.create_runtime()
runtime.run()
print(control_node.stored_data**2 - control_node2.stored_data)
# train_data, vali_data, test_data = dataset()
