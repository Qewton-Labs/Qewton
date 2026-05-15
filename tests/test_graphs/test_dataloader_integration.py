import torch
from pioneer.config.axes import BatchAxes, FeatureAxes, AxesDim
from pioneer.config.data_configurations import DataConfiguration
from pioneer.data.datasets.array_data.base import ArrayLikeDataSet
from pioneer.algorithms.building_blocks.math import Square
from pioneer.data.dataloaders.base import DataLoader
from pioneer.graphs.graphs import Graph
from pioneer.graphs.nodes import Node


def test_dataloader_to_algorithm_flow():
    # 1. Setup Mock Data (100 samples, 5 features)
    n_samples = 100
    n_features = 5
    batch_size = 20
    raw_data = torch.rand(n_samples, n_features)

    # 2. Define Data Configuration
    # DataLoader requires BatchAxes as the first axis for splitting logic
    data_config = DataConfiguration(
        BatchAxes(AxesDim(n_samples)), FeatureAxes(shape=(n_features,))
    )

    # 3. Create Dataset and DataLoader
    dataset = ArrayLikeDataSet(raw_data, data_config)
    dataloader = DataLoader(
        data_set=dataset, batch_size=batch_size, shuffle_data=True, shuffle_seed=42
    )

    # 4. Create Algorithm Node
    algo_node = Square(name="Squarer")

    # 5. Build Graph and Connect
    graph = Graph()
    # Connect the first output port of the loader to the first input of the algo
    graph.connect(dataloader.output_ports[0], algo_node.input_ports[0])

    # 6. Run Execution
    graph.setup()
    graph.run()

    # 7. Assertions
    output_value = algo_node.output_ports[0].value
    assert output_value is not None, "Output port should contain data after run"
    assert output_value.shape == (
        batch_size,
        n_features,
    ), f"Expected batch size {batch_size}, got {output_value.shape[0]}"

    # Verify logic: The output should be the square of the input batch
    input_batch = dataloader.output_ports[0].value
    assert torch.allclose(
        output_value, input_batch**2
    ), "Algorithm logic was not applied correctly to the batch"

    print("Manual connection integration test passed successfully!")


def test_dataloader_tracking_flow():
    # 1. Setup Mock Data
    n_samples = 100
    n_features = 5
    batch_size = 20
    raw_data = torch.rand(n_samples, n_features)

    # 2. Define Data Configuration
    data_config = DataConfiguration(
        BatchAxes(AxesDim(n_samples)), FeatureAxes(shape=(n_features,))
    )

    # 3. Create Dataset and DataLoader
    dataset = ArrayLikeDataSet(raw_data, data_config)
    dataloader = DataLoader(
        data_set=dataset, batch_size=batch_size, shuffle_data=True, shuffle_seed=42
    )

    # 4. Create Algorithm Node
    algo_node = Square(name="Squarer")

    # 5. Build Graph using Tracking Mode
    graph = Graph()
    with graph.tracker():
        # In tracking mode, calling the dataloader returns a TrackingObject
        # representing its output port.
        data_out = dataloader()
        # Calling the algorithm node with that TrackingObject establishes the connection.
        _ = algo_node(data_out)

    # 6. Run Execution
    graph.setup()
    graph.run()

    # 7. Assertions
    output_value = algo_node.output_ports[0].value
    assert output_value is not None, "Output port should contain data after run"
    assert output_value.shape == (batch_size, n_features)

    input_batch = dataloader.output_ports[0].value
    assert torch.allclose(output_value, input_batch**2)

    print("Tracking integration test passed successfully!")


if __name__ == "__main__":
    test_dataloader_to_algorithm_flow()
    test_dataloader_tracking_flow()
