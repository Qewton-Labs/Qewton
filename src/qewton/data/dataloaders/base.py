"""
Base classes for data loading and node-based data sampling in the graph.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING

from qewton.graphs.nodes import NodeState
from qewton.backends import Backend, DEFAULT_DL_BACKEND, TensorType
from qewton.config.devices import Device, cpu

from qewton.optim.base import EvaluationPhase
from qewton.optim.parameters.helpers import HyperParameterState
from qewton.optim.parameters.hyperparameter_base import HyperParameter
from qewton.optim.parameters.number_hyperparameter import (
    DiscreteHyperparameter,
)
from qewton.optim.parameters.categorical_hyperparameter import (
    CategoricalHyperparameter,
)
from qewton.graphs.nodes import Node

if TYPE_CHECKING:
    # Deferred - qewton.visualization transitively imports qewton.optim,
    # which imports GraphBasedTrainer, and qewton.visualization.plots.graph
    # itself imports DataNode from this module, so a module-level import
    # here would be circular.
    from qewton.visualization.axes import ControlAxis
    from qewton.visualization.layout import Layout
    from qewton.visualization.plots.data import DataPlot


class PhaseBatchSize(HyperParameter):
    """The batch size of a DataNode in its current EvaluationPhase, see
    :attr:`DataNode.current_batch_size`. A dim bound to it (e.g. the batch
    axes of the outputs) follows the phase. It is derived from the batch
    sizes of the node, so it is neither tuned nor set.

    Args:
        node (DataNode): The node whose batch size this is.
    """

    def __init__(self, node: "DataNode") -> None:
        self.node = node
        super().__init__(
            parameter_range=(None, None),
            initial_value=None,
            state=HyperParameterState.FIXED,
            name="current_batch_size",
        )

    @property
    def value(self) -> int:
        return self.node.current_batch_size

    def set_value(self, new_value):
        raise RuntimeError(
            "The batch size of the current phase follows the phase; set "
            "batch_size, validation_batch_size or test_batch_size instead."
        )


class DataNode(Node[TensorType]):
    """
    Creates a DataNode which task is to generate/load data for evaluation
    in the graph. This is a base class and should be subclassed for
    specific data loading implementations.

    Args:
        batch_size (int | DiscreteHyperparameter | CategoricalHyperparameter):
            Number of samples per batch in training, and in every phase
            without a batch size of its own.
        name (str, optional): Display name of the node. Defaults to "DataNode".
        state (NodeState, optional): Initial state of the node.
            Defaults to NodeState.FIXED.
        backend (type[Backend] | None, optional): Computing backend
            (e.g., TorchBackend). Defaults to DEFAULT_DL_BACKEND.
        validation_batch_size (int | DiscreteHyperparameter |
            CategoricalHyperparameter | None, optional): Number of samples
            per batch in validation. Defaults to None, i.e. `batch_size`.
        test_batch_size (int | DiscreteHyperparameter |
            CategoricalHyperparameter | None, optional): Number of samples
            per batch in testing. Defaults to None, i.e. `batch_size`.
    """

    def __init__(
        self,
        batch_size: int | DiscreteHyperparameter | CategoricalHyperparameter,
        name: str = "DataNode",
        state: NodeState = NodeState.FIXED,
        backend: type[Backend[TensorType]] | None = DEFAULT_DL_BACKEND,
        validation_batch_size: (
            int | DiscreteHyperparameter | CategoricalHyperparameter | None
        ) = None,
        test_batch_size: (
            int | DiscreteHyperparameter | CategoricalHyperparameter | None
        ) = None,
    ) -> None:
        self._batch_size = HyperParameter.from_value(batch_size, name="batch_size")
        # A phase without a batch size of its own shares the training
        # HyperParameter, so it follows changes of it (e.g. while tuning).
        self._validation_batch_size = (
            self._batch_size
            if validation_batch_size is None
            else HyperParameter.from_value(
                validation_batch_size, name="validation_batch_size"
            )
        )
        self._test_batch_size = (
            self._batch_size
            if test_batch_size is None
            else HyperParameter.from_value(test_batch_size, name="test_batch_size")
        )
        self._phase_batch_size = PhaseBatchSize(self)
        self._batch_progress = 0
        self._is_cached = False
        self._device = cpu
        if backend is None:
            backend = Backend
        super().__init__(name, state, backend=backend)

    @property
    def batch_size(self) -> int:
        """The number of samples per batch in training."""
        return self._batch_size.value

    @property
    def validation_batch_size(self) -> int:
        """The number of samples per batch in validation; `batch_size` if
        none was given."""
        return self._validation_batch_size.value

    @property
    def test_batch_size(self) -> int:
        """The number of samples per batch in testing; `batch_size` if none
        was given."""
        return self._test_batch_size.value

    @property
    def current_batch_size(self) -> int:
        """The number of samples per batch in the current mode: the
        validation or test batch size in those phases, else `batch_size`."""
        if self.mode == EvaluationPhase.VALIDATION:
            return self.validation_batch_size
        if self.mode == EvaluationPhase.TEST:
            return self.test_batch_size
        return self.batch_size

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def training_batches(self) -> int:
        return len(self)

    @property
    def is_cached(self):
        return self._is_cached

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        parameters = [
            self._batch_size,
            self._validation_batch_size,
            self._test_batch_size,
        ]
        # Each shared HyperParameter once; by id, since == builds a condition.
        return list({id(p): p for p in parameters}.values())

    def cache(self, n_batches: int = -1):
        pass

    def clear_cache(self):
        pass

    def provides_data_in_phase(
        self, phase: EvaluationPhase  # pylint: disable=unused-argument
    ) -> bool:
        """Checks whether this node can provide data in the EvaluationPhase.

        Args:
            phase (EvaluationPhase): The evaluation phase to check for
                data provision capability.

        Returns:
            bool: If True, the node can provide data in the specified phase,
                otherwise, False.
        """
        return False

    def to(self, device: str | Device):
        """Move the data node to the specified device.

        Args:
            device (str, Device): The device to move the data node to
                (e.g., 'cpu', 'cuda').
        """
        self._device = device

    def visualize(
        self,
        mode: EvaluationPhase = EvaluationPhase.VALIDATION,
        plot_type: "type[DataPlot] | None" = None,
        controls: "list[ControlAxis]" = (),
        default_control: "type[ControlAxis] | None" = None,
        **plot_kwargs,
    ) -> "Layout":
        """Run this node on its own (a DataNode is always a graph source,
        so it needs no Graph) and build a layout showing every output port.

        Args:
            mode: The EvaluationPhase this node is run in; it determines
                which split supplies the batch.
            plot_type: Passed to
                :func:`~qewton.visualization.auto.auto_plot` for every port.
            controls: Control axes for chosen dimensions, passed to
                auto_plot for every port that has them.
            default_control: Control axis class for every other dimension
                not drawn; None for auto_plot's default. Every control is
                shared by all ports whose data has its axes or Variable (see
                :class:`~qewton.visualization.auto.AutoPlots`).
            **plot_kwargs: Passed to auto_plot for every port.

        Returns:
            Layout: The plots arranged by
                :func:`~qewton.visualization.layout.auto_layout`.

        Raises:
            ValueError: If this node provides no data in `mode`.
        """
        from qewton.visualization.auto import AutoPlots
        from qewton.visualization.layout import auto_layout

        if not self.provides_data_in_phase(mode):
            raise ValueError(
                f"{self} provides no data in {mode} - pass a mode whose "
                "split/dataset is non-empty (see provides_data_in_phase())."
            )

        self.set_mode(mode)
        values = self()
        if len(self.output_ports) == 1:
            values = (values,)

        if default_control is not None:
            plot_kwargs["default_control"] = default_control
        plots = AutoPlots(controls)
        for port, value in zip(self.output_ports, values):
            plots.add(
                self.backend.to_numpy(value),
                port.data_configuration,
                plot_type,
                **plot_kwargs,
            )
        return auto_layout(plots.plots)
