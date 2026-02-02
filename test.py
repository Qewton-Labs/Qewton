import numpy as np
import pioneer


data = np.random.random((100, 10, 10, 2))

X = pioneer.data.Variable("x", 2)
dataset = pioneer.data.DataSet.from_data(data, X, batch_size=100)

# print(dataset.ports)

control_node = pioneer.pipeline.ControlNode(dataset.data_config, "SaveNode1")

pipeline = pioneer.pipeline.Pipeline()

pipeline.connect(dataset.output_ports["test_data"], control_node.input_ports["input"])
pipeline.validate()
runtime = pipeline.create_runtime()
runtime.run()
print(control_node.stored_data)
train_data, vali_data, test_data = dataset()


# from typing import TypedDict


# class BaseData(TypedDict):
#     pass


# class ChildAData(BaseData):
#     A: int
#     B: int


# class Test1:

#     def get_ports(self) -> BaseData:
#         return BaseData()


# class Test2:

#     def get_ports(self) -> ChildAData:
#         return {"A": 1, "B": 2}


# test_2 = Test2()
# test_2.get_ports()[]
