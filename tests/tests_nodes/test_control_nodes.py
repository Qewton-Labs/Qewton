import pytest
from src.pioneer.nodes.control_nodes import ControlNode
from src.pioneer.config.configuration_base import DataConfiguration
from src.pioneer.config.axis import BatchAxis, FeatureAxis


class TestControlNodeInit:
    """Tests for ControlNode initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        assert node.data_config == config
        assert node.name == "ControlNode"
        assert node.stored_data is None

    def test_init_custom_name(self):
        """Test initialization with custom name."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config, name="CustomControl")
        assert node.name == "CustomControl"


class TestControlNodePorts:
    """Tests for ControlNode ports."""

    def test_input_and_output_ports_same(self):
        """Test that input and output ports reference the same port object."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        input_port = node.input_ports[node.InputKeys.INPUT]
        output_port = node.output_ports[node.OutputKeys.OUTPUT]
        assert input_port == output_port

    def test_port_is_required(self):
        """Test that the port is marked as required."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        port = node.input_ports[node.InputKeys.INPUT]
        assert port.required is True


class TestControlNodeRun:
    """Tests for ControlNode.run method."""

    def test_run_stores_and_returns_data(self):
        """Test that run stores input and returns it as output."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        test_data = [1, 2, 3, 4, 5]
        outputs = node.run({"input": test_data})

        assert node.stored_data == test_data
        assert outputs["output"] == test_data

    def test_run_with_none_inputs_raises_error(self):
        """Test that run raises ValueError when inputs is None."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        with pytest.raises(RuntimeError):
            node.run(None)

    def test_run_overwrites_previous_data(self):
        """Test that run overwrites previously stored data."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        node.run({"input": [1, 2, 3]})
        assert node.stored_data == [1, 2, 3]

        node.run({"input": [4, 5, 6]})
        assert node.stored_data == [4, 5, 6]

    def test_run_with_empty_dict_raises_error(self):
        """Test that run with empty dict raises KeyError."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        with pytest.raises(KeyError):
            node.run({})


class TestControlNodeReset:
    """Tests for ControlNode.reset method."""

    def test_reset_clears_stored_data(self):
        """Test that reset clears stored data."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        node.run({"input": [1, 2, 3]})
        assert node.stored_data == [1, 2, 3]

        node.reset()
        assert node.stored_data is None

    def test_reset_on_empty_node(self):
        """Test that reset works on node with no stored data."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = ControlNode(config)
        node.reset()  # Should not raise error
        assert node.stored_data is None
