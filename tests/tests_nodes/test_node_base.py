from unittest.mock import Mock
import pytest
from src.pioneer.nodes.base import Port, Node, _NodeRuntime
from src.pioneer.config.configuration_base import DataConfiguration
from src.pioneer.config.axis import BatchAxis, FeatureAxis
from src.pioneer.config.variables import Variable
from src.pioneer.optim.hyperparameter.base import (
    HyperParameter,
    ContinuousHyperparameter,
)
from src.pioneer.optim.base import EvaluationMode


class TestPortInit:
    """Tests for Port.__init__ method."""

    def test_port_init_basic(self):
        """Test basic port initialization."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = Mock(spec=Node)
        port = Port(config, node, "test_port")

        assert port.data_configuration == config
        assert port.node == node
        assert port.name == "test_port"
        assert port.required is False

    def test_port_init_required(self):
        """Test port initialization with required=True."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = Mock(spec=Node)
        port = Port(config, node, "required_port", is_required=True)

        assert port.required is True

    def test_port_init_with_none_dtype(self):
        """Test port with None dtype."""
        config = DataConfiguration(
            None, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = Mock(spec=Node)
        port = Port(config, node, "port")
        assert port.data_configuration.dtype is None


class TestPortEquality:
    """Tests for Port.__eq__ method."""

    def test_port_eq_identical(self):
        """Test equality of identical ports."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = Mock(spec=Node)
        port1 = Port(config, node, "port")
        port2 = Port(config, node, "port")
        assert port1 == port2

    def test_port_eq_different_config(self):
        """Test inequality with different configurations."""
        config1 = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        config2 = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=20)], FeatureAxis(size=20)
        )
        node = Mock(spec=Node)
        port1 = Port(config1, node, "port")
        port2 = Port(config2, node, "port")
        assert port1 != port2

    def test_port_eq_different_node(self):
        """Test inequality with different nodes."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node1 = Mock(spec=Node)
        node2 = Mock(spec=Node)
        port1 = Port(config, node1, "port")
        port2 = Port(config, node2, "port")
        assert port1 != port2

    def test_port_eq_different_name(self):
        """Test inequality with different names."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = Mock(spec=Node)
        port1 = Port(config, node, "port1")
        port2 = Port(config, node, "port2")
        assert port1 != port2

    def test_port_eq_non_port_object(self):
        """Test equality with non-Port object returns False."""
        config = DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        node = Mock(spec=Node)
        port = Port(config, node, "port")
        assert port != "not a port"
        assert port != 42
        assert port is not None


class ConcreteNode(Node):
    """Concrete implementation of Node for testing."""

    def __init__(self, name: str = "TestNode", input_config=None, output_config=None):
        super().__init__(name)
        self._input_config = input_config or DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )
        self._output_config = output_config or DataConfiguration(
            float, [BatchAxis(), FeatureAxis(size=10)], FeatureAxis(size=10)
        )

    @property
    def input_ports(self) -> dict[str, Port]:
        return {
            self.InputKeys.INPUT: Port(
                self._input_config, self, self.InputKeys.INPUT, is_required=True
            )
        }

    @property
    def output_ports(self) -> dict[str, Port]:
        return {
            self.OutputKeys.OUTPUT: Port(
                self._output_config, self, self.OutputKeys.OUTPUT
            )
        }

    def run(self, inputs: dict | None = None) -> dict:
        if inputs is None:
            inputs = {}
        return {self.OutputKeys.OUTPUT: inputs.get(self.InputKeys.INPUT, [])}


class TestNodeInit:
    """Tests for Node.__init__ method."""

    def test_node_init_default_name(self):
        """Test node initialization with default name."""
        node = ConcreteNode()
        assert node.name == "TestNode"

    def test_node_init_custom_name(self):
        """Test node initialization with custom name."""
        node = ConcreteNode(name="CustomNode")
        assert node.name == "CustomNode"

    def test_node_init_mode_default(self):
        """Test node initialization mode is ALWAYS."""
        node = ConcreteNode()
        assert node.mode == EvaluationMode.ALWAYS


class TestNodePorts:
    """Tests for Node input and output ports."""

    def test_input_ports_returns_dict(self):
        """Test that input_ports returns a dictionary."""
        node = ConcreteNode()
        ports = node.input_ports
        assert isinstance(ports, dict)

    def test_output_ports_returns_dict(self):
        """Test that output_ports returns a dictionary."""
        node = ConcreteNode()
        ports = node.output_ports
        assert isinstance(ports, dict)

    def test_input_port_required(self):
        """Test that default input port is required."""
        node = ConcreteNode()
        port = node.input_ports[Node.InputKeys.INPUT]
        assert port.required is True

    def test_output_port_not_required(self):
        """Test that output port is not required."""
        node = ConcreteNode()
        port = node.output_ports[Node.OutputKeys.OUTPUT]
        assert port.required is False


class TestNodeCall:
    """Tests for Node.__call__ method."""

    def test_call_with_positional_args(self):
        """Test calling node with positional arguments."""
        node = ConcreteNode()
        result = node([1, 2, 3])
        assert result == [1, 2, 3]

    def test_call_with_keyword_args(self):
        """Test calling node with keyword arguments."""
        node = ConcreteNode()
        result = node(input=[4, 5, 6])
        assert result == [4, 5, 6]

    def test_call_returns_single_value_unwrapped(self):
        """Test that single output is unwrapped."""
        node = ConcreteNode()
        result = node([1, 2])
        assert not isinstance(result, tuple)

    def test_call_missing_required_input_raises_error(self):
        """Test that missing required input raises ValueError."""
        node = ConcreteNode()
        with pytest.raises(ValueError, match="Missing input"):
            node()


class TestNodeBindInputs:
    """Tests for Node._bind_inputs method."""

    def test_bind_inputs_positional_only(self):
        """Test binding positional arguments."""
        node = ConcreteNode()
        inputs = node._bind_inputs([1, 2, 3])  # pylint: disable=protected-access
        assert inputs[Node.InputKeys.INPUT] == [1, 2, 3]

    def test_bind_inputs_keyword_only(self):
        """Test binding keyword arguments."""
        node = ConcreteNode()
        inputs = node._bind_inputs(input=[4, 5, 6])  # pylint: disable=protected-access
        assert inputs[Node.InputKeys.INPUT] == [4, 5, 6]

    def test_bind_inputs_positional_and_keyword(self):
        """Test binding both positional and keyword arguments."""
        node = ConcreteNode()
        inputs = node._bind_inputs(  # pylint: disable=protected-access
            [1, 2], input=[3, 4]
        )
        # Keyword should override positional
        assert inputs[Node.InputKeys.INPUT] == [3, 4]

    def test_bind_inputs_missing_required_raises_error(self):
        """Test that missing required input raises ValueError."""
        node = ConcreteNode()
        with pytest.raises(ValueError, match="Missing input"):
            node._bind_inputs()  # pylint: disable=protected-access

    def test_bind_inputs_too_many_positional_ignored(self):
        """Test that extra positional arguments are ignored."""
        node = ConcreteNode()
        inputs = node._bind_inputs([1], [2], [3])  # pylint: disable=protected-access
        assert inputs[Node.InputKeys.INPUT] == [1]


class TestNodeRun:
    """Tests for Node.run method."""

    def test_run_with_inputs(self):
        """Test running node with inputs."""
        node = ConcreteNode()
        outputs = node.run({Node.InputKeys.INPUT: [1, 2, 3]})
        assert outputs[Node.OutputKeys.OUTPUT] == [1, 2, 3]

    def test_run_with_none_inputs(self):
        """Test running node with None inputs."""
        node = ConcreteNode()
        outputs = node.run(None)
        assert outputs[Node.OutputKeys.OUTPUT] == []


class TestNodeGetitem:
    """Tests for Node.__getitem__ method."""

    def test_getitem_input_port_by_name(self):
        """Test accessing input port by name."""
        node = ConcreteNode()
        port = node["input"]
        assert port == node.input_ports["input"]

    def test_getitem_output_port_by_name(self):
        """Test accessing output port by name."""
        node = ConcreteNode()
        port = node["output"]
        assert port == node.output_ports["output"]

    def test_getitem_by_variable(self):
        """Test accessing port by Variable."""
        node = ConcreteNode()
        var = Variable(name="input", dim=1)
        port = node[var]
        assert port == node.input_ports["input"]

    def test_getitem_variable_multiple_keys_raises_error(self):
        """Test that Variable with multiple keys raises AssertionError."""
        node = ConcreteNode()
        var = Variable.from_dict({"input": 1, "output": 1})
        with pytest.raises(AssertionError):
            _ = node[var]

    def test_getitem_nonexistent_port_raises_error(self):
        """Test that nonexistent port raises ValueError."""
        node = ConcreteNode()
        with pytest.raises(ValueError, match="Port .* does not exist"):
            _ = node["nonexistent"]


class TestNodeHyperparameters:
    """Tests for Node.hyperparameters property."""

    def test_hyperparameters_empty(self):
        """Test node with no hyperparameters."""
        node = ConcreteNode()
        assert node.hyperparameters == []

    def test_hyperparameters_single(self):
        """Test node with single hyperparameter."""

        class NodeWithHP(ConcreteNode):
            def __init__(self):
                super().__init__()
                self.learning_rate = ContinuousHyperparameter(
                    (0.001, 0.1), initial_value=0.01, name="lr"
                )

        node = NodeWithHP()
        assert len(node.hyperparameters) == 1
        assert isinstance(node.hyperparameters[0], HyperParameter)

    def test_hyperparameters_multiple(self):
        """Test node with multiple hyperparameters."""

        class NodeWithMultipleHP(ConcreteNode):
            def __init__(self):
                super().__init__()
                self.lr = ContinuousHyperparameter((0.001, 0.1), name="lr")
                self.dropout = ContinuousHyperparameter((0.0, 0.5), name="dropout")

        node = NodeWithMultipleHP()
        assert len(node.hyperparameters) == 2

    def test_hyperparameters_excludes_non_hyperparameters(self):
        """Test that non-hyperparameter attributes are excluded."""

        class NodeWithMixed(ConcreteNode):
            def __init__(self):
                super().__init__()
                self.hp = ContinuousHyperparameter((0.001, 0.1), name="hp")
                self.other_value = 42

        node = NodeWithMixed()
        assert len(node.hyperparameters) == 1
        assert all(isinstance(h, HyperParameter) for h in node.hyperparameters)


class TestNodeRuntime:
    """Tests for _NodeRuntime class."""

    def test_runtime_init(self):
        """Test runtime initialization."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        assert runtime.node == node
        assert not runtime.received_inputs
        assert runtime.has_run is False

    def test_runtime_receive(self):
        """Test receiving input."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        runtime.receive("input", [1, 2, 3])
        assert runtime.received_inputs["input"] == [1, 2, 3]

    def test_runtime_receive_multiple(self):
        """Test receiving multiple inputs."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        runtime.receive("input1", [1, 2])
        runtime.receive("input2", [3, 4])
        assert runtime.received_inputs["input1"] == [1, 2]
        assert runtime.received_inputs["input2"] == [3, 4]

    def test_runtime_is_ready_all_required_received(self):
        """Test is_ready when all required inputs received."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        runtime.receive("input", [1, 2, 3])
        assert runtime.is_ready() is True

    def test_runtime_is_ready_missing_required(self):
        """Test is_ready when required input missing."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        assert runtime.is_ready() is False

    def test_runtime_is_ready_already_run(self):
        """Test is_ready returns False if already run."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        runtime.receive("input", [1, 2])
        runtime.has_run = True
        assert runtime.is_ready() is False

    def test_runtime_run_success(self):
        """Test successful run."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        runtime.receive("input", [1, 2, 3])
        outputs = runtime.run()
        assert outputs["output"] == [1, 2, 3]

    def test_runtime_run_clears_inputs(self):
        """Test that run clears received inputs."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        runtime.receive("input", [1, 2, 3])
        runtime.run()
        assert not runtime.received_inputs

    def test_runtime_run_sets_has_run(self):
        """Test that run sets has_run to True."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        runtime.receive("input", [1, 2])
        runtime.run()
        assert runtime.has_run is True

    def test_runtime_run_not_ready_raises_error(self):
        """Test that run raises RuntimeError when not ready."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        with pytest.raises(RuntimeError, match="not ready"):
            runtime.run()

    def test_runtime_run_missing_input_raises_error(self):
        """Test that run raises RuntimeError with missing required input."""
        node = ConcreteNode()
        runtime = _NodeRuntime(node)
        with pytest.raises(RuntimeError):
            runtime.run()


class TestNodeIntegration:
    """Integration tests for Node and Port."""

    def test_node_create_runtime(self):
        """Test creating runtime from node."""
        node = ConcreteNode()
        runtime = node.create_runtime()
        assert isinstance(runtime, _NodeRuntime)
        assert runtime.node == node

    def test_full_node_execution_flow(self):
        """Test complete node execution flow."""
        node = ConcreteNode()
        runtime = node.create_runtime()

        # Prepare inputs
        test_data = [1, 2, 3, 4, 5]
        runtime.receive("input", test_data)

        # Execute
        assert runtime.is_ready()
        outputs = runtime.run()

        # Verify
        assert outputs["output"] == test_data
        assert runtime.has_run is True

    def test_node_getitem_chain_with_input(self):
        """Test chaining getitem with node methods."""
        node = ConcreteNode()
        port = node["input"]
        assert isinstance(port, Port)
        assert port.required is True

    def test_multiple_nodes_independent_runtimes(self):
        """Test that multiple nodes have independent runtimes."""
        node1 = ConcreteNode(name="Node1")
        node2 = ConcreteNode(name="Node2")

        runtime1 = node1.create_runtime()
        runtime2 = node2.create_runtime()

        runtime1.receive("input", [1, 2])
        assert not runtime2.received_inputs

    def test_node_with_variable_input_access(self):
        """Test accessing node ports using Variable."""
        node = ConcreteNode()
        var_input = Variable(name="input", dim=1)
        var_output = Variable(name="output", dim=1)

        port_in = node[var_input]
        port_out = node[var_output]

        assert port_in == node.input_ports["input"]
        assert port_out == node.output_ports["output"]

    def test_empty_variable_list_in_getitem_raises_error(self):
        """Test that empty Variable raises AssertionError."""
        node = ConcreteNode()
        empty_var = Variable()
        with pytest.raises(AssertionError):
            _ = node[empty_var]
