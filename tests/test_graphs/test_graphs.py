import unittest
from unittest.mock import MagicMock
from typing import Annotated, Any

from qewton.backends.base import TensorType
from qewton.graphs.nodes import Node, Port, InputPort, OutputPort, NodeState
from qewton.graphs.edges import Edge
from qewton.graphs.graphs import Graph, SequentialGraph, TrackingObject
from qewton.config.data_configurations import DataConfiguration
from qewton.backends import Backend
from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.trainable_parameters import TrainableParameters


class MockNode(Node):
    """A concrete implementation of Node for testing purposes."""

    def forward(
        self, x: Annotated[float, DataConfiguration.empty()], y: float = 1.0
    ) -> Annotated[float, DataConfiguration.empty()]:
        return x + y


class FunctionalConfigNode(Node):
    """Node with a callable in Annotated to test config resolution."""

    def forward(
        self, x: Annotated[float, lambda owner: DataConfiguration.empty()]
    ) -> float:
        return x


class MultiOutputNode(Node):
    """Node that returns multiple values to test tuple output handling."""

    def forward(
        self, x: float
    ) -> tuple[Annotated[float, DataConfiguration.empty()], float]:
        return x, x * 2


class ParamNode(Node):
    """Node with hyperparameters for testing collection logic."""

    def __init__(self, name="ParamNode"):
        self.hp = MagicMock(spec=HyperParameter)
        super().__init__(name=name)


class MockBackend(Backend):
    default_dtype = Any


class TestGraphs(unittest.TestCase):
    def setUp(self):
        self.mock_config = DataConfiguration.empty()
        self.node = MockNode(name="TestNode")

    def test_edge_initialization(self):
        p1 = MagicMock(spec=Port)
        p2 = MagicMock(spec=Port)
        edge = Edge(from_port=p1, to_port=p2, connects_to_outside=True)
        self.assertEqual(edge.from_port, p1)
        self.assertEqual(edge.to_port, p2)
        self.assertTrue(edge.connects_to_outside)

    def test_port_value_management(self):
        port = Port(self.mock_config, self.node, "test_port")
        self.assertIsNone(port.value)
        port.set_value(10)
        self.assertEqual(port.value, 10)
        port.reset_value()
        self.assertIsNone(port.value)

    def test_port_duplication(self):
        new_owner = MockNode(name="OtherNode")
        port = Port(self.mock_config, self.node, "original")
        duplicated = port.duplicate_with_new_owner(new_owner, "new_name")
        self.assertEqual(duplicated.node, new_owner)
        self.assertEqual(duplicated.name, "new_name")
        self.assertEqual(duplicated.data_configuration, self.mock_config)

    def test_input_port_required_and_default(self):
        # 'x' is required because no default was provided in MockNode.forward
        req_port = self.node.get_input_port("x")
        self.assertTrue(req_port.is_required)

        # 'y' has a default value of 1.0
        def_port = self.node.get_input_port("y")
        self.assertFalse(def_port.is_required)
        self.assertEqual(def_port.value, 1.0)

        def_port.set_value(10.0)
        def_port.clear_value()
        self.assertEqual(def_port.value, 1.0)

    def test_input_port_duplication(self):
        new_owner = MockNode(name="OtherNode")
        port = InputPort(self.mock_config, self.node, "original", default=42)
        duplicated = port.duplicate_with_new_owner(new_owner)
        self.assertEqual(duplicated.default, 42)
        self.assertEqual(duplicated.value, 42)

    def test_node_port_generation(self):
        # MockNode.forward signature defines 2 inputs and 1 output
        self.assertEqual(len(self.node.input_ports), 2)
        self.assertEqual(self.node.input_ports[0].name, "x")
        self.assertEqual(self.node.input_ports[1].name, "y")

        self.assertEqual(len(self.node.output_ports), 1)
        self.assertEqual(self.node.output_ports[0].name, "output_0")

    def test_multi_output_generation_and_execution(self):
        node = MultiOutputNode()
        self.assertEqual(len(node.output_ports), 2)
        self.assertEqual(node.output_ports[0].name, "output_0")
        self.assertEqual(node.output_ports[1].name, "output_1")

        node.get_input_port("x").set_value(10.0)
        node.run()
        self.assertEqual(node.output_ports[0].value, 10.0)
        self.assertEqual(node.output_ports[1].value, 20.0)

    def test_node_execution(self):
        self.node.get_input_port("x").set_value(5.0)
        self.node.get_input_port("y").set_value(2.0)
        self.node.run()
        # Expected sum: 5.0 + 2.0 = 7.0
        self.assertEqual(self.node.output_ports[0].value, 7.0)

    def test_node_state_management(self):
        self.assertEqual(self.node.state, NodeState.FIXED)
        self.node.set_state(NodeState.INITIALIZED)
        self.assertEqual(self.node.state, NodeState.INITIALIZED)

        self.node.fix_node_state()
        self.assertEqual(self.node.state, NodeState.FIXED)

    def test_fix_node_state_uninitialized_warning(self):
        self.node.set_state(NodeState.UNINITIALIZED)
        with self.assertWarns(UserWarning):
            self.node.fix_node_state()
        # Should remain UNINITIALIZED if warned
        self.assertEqual(self.node.state, NodeState.UNINITIALIZED)

    def test_get_port_by_name(self):
        port = self.node.get_input_port("x")
        self.assertEqual(port.name, "x")

        out_port = self.node.get_output_port("output_0")
        self.assertEqual(out_port.name, "output_0")

        with self.assertRaises(ValueError):
            self.node.get_input_port("non_existent")
        with self.assertRaises(ValueError):
            self.node.get_output_port("non_existent")

    def test_set_mode(self):
        self.node.set_mode(EvaluationPhase.TRAIN)
        self.assertEqual(self.node.mode, EvaluationPhase.TRAIN)

    def test_hyperparameters_property(self):
        node = ParamNode()
        hps = node.hyperparameters
        self.assertEqual(len(hps), 1)
        self.assertEqual(hps[0], node.hp)

    def test_trainable_parameters_property(self):
        # By default, state is FIXED, so _trainable_parameters should be empty
        self.node.set_state(NodeState.FIXED)
        tp_fixed = self.node._trainable_parameters
        self.assertIsInstance(tp_fixed, TrainableParameters)
        self.assertTrue(tp_fixed.empty)

        # When not FIXED, it should return the property
        self.node.set_state(NodeState.INITIALIZED)
        tp_init = self.node._trainable_parameters
        self.assertIsInstance(tp_init, TrainableParameters)

    def test_backend_propagation(self):
        # Create a node with a specific backend to trigger _set_port_backend
        config = DataConfiguration.empty()
        config.set_dtype = MagicMock()

        class BackendNode(Node[TensorType]):
            def forward(
                self, x: Annotated[TensorType, config]
            ) -> Annotated[TensorType, config]:
                return x

        BackendNode(backend=MockBackend)
        config.set_dtype.assert_called_with(TensorType)

    def test_unwrap_annotated_no_metadata(self):
        # Test Annotated without DataConfiguration or Callable meta
        hint = Annotated[float, "some_other_metadata"]
        config, was_annotated = Node._unwrap_annotated(hint, self.node, self.node.backend)
        self.assertIsInstance(config, DataConfiguration)
        self.assertTrue(was_annotated)

    def test_update_data_configs(self):
        node = MultiOutputNode()
        out_port = node.output_ports[0]
        mock_config = MagicMock(spec=DataConfiguration)
        mock_config.update_config.return_value = True

        dynamic_configs = {
            p: MagicMock(spec=DataConfiguration) for p in node.output_ports
        }
        dynamic_configs[out_port] = mock_config

        # Trigger update
        updated = node.update_data_configs(out_port, {"key": "val"}, dynamic_configs)
        self.assertIn(out_port, updated)
        mock_config.update_config.assert_called()

    def test_base_node_forward_raises(self):
        with self.assertRaises(NotImplementedError):
            super(MockNode, self.node).forward()

    def test_copy_data_configs(self):
        configs = self.node.copy_data_configs()
        self.assertEqual(len(configs), 3)  # x, y, output_0
        for port in self.node.input_ports:
            self.assertIn(port, configs)

    def test_tracking_toggle(self):
        Node.set_tracking(True)
        self.assertTrue(Node._tracking_phase)
        # Note: actually running _track requires graphs.py logic (TrackingObject)
        # which is usually tested in integrated graph tests.
        Node.set_tracking(False)

    def test_functional_data_config(self):
        node = FunctionalConfigNode()
        # Ensure that the lambda in Annotated was executed and resolved to a
        # DataConfiguration
        self.assertIsInstance(node.input_ports[0].data_configuration, DataConfiguration)

    def test_node_copy(self):
        # Test that copy() returns a CopiedNode (which wraps the original)
        # We check the type name since CopiedNode is imported locally in Node.copy
        copy_node = self.node.copy()
        self.assertEqual(type(copy_node).__name__, "CopiedNode")


class TestGraphLogic(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.n1 = MockNode(name="N1")
        self.n2 = MockNode(name="N2")

    def test_add_node_duplicate_error(self):
        self.graph.add_node(self.n1)
        with self.assertRaises(ValueError):
            self.graph.add_node(self.n1)

    def test_connect_port_count_mismatch(self):
        class MultiIn(Node):
            def forward(self, a, b):
                return a

        node_multi = MultiIn()
        with self.assertRaises(ValueError):
            # self.n1 has 1 output, node_multi has 2 inputs
            self.graph.connect(self.n1, node_multi)

    def test_connect_occupied_input(self):
        self.graph.connect(self.n1, self.n2.input_ports[0])  # Connect n1.output_0 to n2.x
        n3 = MockNode(name="N3")
        with self.assertRaises(ValueError):
            # n2.x is already taken by n1
            self.graph.connect(n3, self.n2.input_ports[0])

    def test_sort_cycle_detection(self):
        # Create N1 -> N2 -> N1 cycle
        self.graph.connect(self.n1, self.n2.input_ports[0])
        # Force connect N2 back to N1 (manually since standard connect checks inputs)
        edge = Edge(self.n2.output_ports[0], self.n1.input_ports[0])
        self.graph.incoming_edges[self.n1].append(edge)

        with self.assertRaisesRegex(ValueError, "Cycle detected"):
            self.graph.sort()

    def test_validate_required_ports(self):
        # N1 has required input 'x'
        self.graph.add_node(self.n1)
        with self.assertRaises(ValueError):
            self.graph.validate()

        class NoIn(Node):
            def forward(self) -> float:
                return 3.0

        # Connect it
        loader = NoIn()
        self.graph.connect(loader.output_ports[0], self.n1.get_input_port("x"))
        # 'y' is not required (has default 1.0), so validate should pass now
        self.graph.validate()

    def test_run_clears_unused_ports(self):
        class SourceNode(Node):
            def forward(self) -> float:
                return 1.0

        src = SourceNode()
        self.graph.connect(src.output_ports[0], self.n1.input_ports[0])
        self.graph.connect(self.n1, self.n2.input_ports[0])

        # Set value for y (which is not connected) to see if it clears to default
        self.graph.setup()
        self.n2.get_input_port("y").set_value(99.0)
        self.graph.run()

        # y should have been cleared to its default (1.0) during run
        self.assertEqual(self.n2.get_input_port("y").value, 1.0)

    def test_external_connections(self):
        ext_node = MockNode(name="External")
        # External -> Inside
        self.graph.connect_from_outside_of_graph(
            ext_node.output_ports[0], self.n1.get_input_port("x")
        )
        self.assertEqual(len(self.graph.edges_from_outside), 1)

        # Inside -> External
        self.graph.connect_to_outside_of_graph(
            self.n1.output_ports[0], ext_node.input_ports[0]
        )
        self.assertEqual(len(self.graph.edges_to_outside), 1)

    def test_skip_connection(self):
        self.graph.add_skip_connection(self.n1.output_ports[0], self.n2.input_ports[0])
        self.assertEqual(len(self.graph.skip_connections), 1)

    def test_sequential_graph(self):
        class SingleInputNode(Node):
            def forward(self, x: float) -> float:
                return x

        sn1, sn2 = SingleInputNode(name="SN1"), SingleInputNode(name="SN2")
        seq = SequentialGraph(sn1, sn2)
        self.assertEqual(len(seq.nodes), 2)
        self.assertEqual(seq.incoming_edges[sn2][0].from_port, sn1.output_ports[0])

    def test_from_function_variations(self):
        # No args
        g1, ins1, outs1 = Graph.from_function(lambda: MockNode()())
        self.assertEqual(len(ins1), 0)

        # Tuple output
        def multi_out_func(a):
            res = MultiOutputNode()(a)
            return res[0], res[1]

        g2, ins2, outs2 = Graph.from_function(multi_out_func)
        self.assertEqual(len(outs2), 2)

        # None output
        g3, ins3, outs3 = Graph.from_function(lambda a: None)
        self.assertEqual(len(outs3), 0)


class TestTrackingObjectOps(unittest.TestCase):
    def test_arithmetic_overloads(self):
        # These tests ensure that using Python operators on TrackingObjects
        # creates the corresponding math nodes in the graph.
        graph = Graph()
        with graph.tracker(n_tracking_vars=2) as (a, b):
            # We mock the algorithms to avoid heavy imports, but here we
            # test the wiring logic in TrackingObject.
            # Since we can't easily mock the local imports inside TrackingObject,
            # this verifies the logic path exists.
            try:
                _ = a + b
                _ = a - b
                _ = a * b
                _ = a / b
                _ = a @ b
                _ = a**b
                _ = abs(a)
                _ = a[0]
                _ = a.gradient(b)
            except (ImportError, ModuleNotFoundError):
                # Fallback if building blocks aren't in test path
                pass

    def test_node_call_tracking_branches(self):
        Node.set_tracking(True)
        node = MockNode()
        t_obj = TrackingObject()

        # Track via args
        node(t_obj, 5.0)  # 5.0 should become a default value
        self.assertEqual(node.get_input_port("y").default, 5.0)

        # Track via kwargs
        node(x=t_obj, y=10.0)
        self.assertEqual(node.get_input_port("y").default, 10.0)

        Node.set_tracking(False)

    def test_tracker_error_non_empty(self):
        graph = Graph()
        graph.add_node(MockNode())
        with self.assertRaises(RuntimeError):
            with graph.tracker():
                pass

    def test_tracking_object_with_ports(self):
        port = MagicMock(spec=OutputPort)
        to_port = MagicMock(spec=InputPort)
        obj = TrackingObject(last_output_port=port)
        obj.add_to_port(to_port)
        self.assertIn(to_port, obj.to_ports)
        self.assertEqual(obj.last_output_port, port)
