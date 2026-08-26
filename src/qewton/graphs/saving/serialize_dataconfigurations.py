from dataclasses import dataclass

from qewton.config.axes import Axes, AxesDim, OperationDim
from qewton.config.data_configurations import DataConfiguration
from qewton.graphs.nodes import NodeConfig

from qewton.graphs.saving.schema import (
    KEY_DATA_CONFIGURATIONS,
    KEY_AXES,
    KEY_AXES_DIMENSIONS,
)


@dataclass
class DataConfigSerializationResult:
    serialized_payload: dict
    config_list: list[DataConfiguration]
    axes_list: list[Axes]
    axes_dim_list: list[AxesDim]


def serialize_data_configurations(
    node_config: NodeConfig,
) -> DataConfigSerializationResult:
    # Find all unique axes and axes dimensions from the node's ports
    config_list = []
    axes_set: set[Axes] = set()
    axes_dim_set: set[AxesDim] = set()

    for port in node_config.input_ports + node_config.output_ports:
        config_list.append(port.data_configuration)
        for axes in port.data_configuration.axes:
            axes_set.add(axes)
            for dim in axes.shape:
                axes_dim_set.add(dim)

    axes_dim_list: list[AxesDim] = order_axes_dims(axes_dim_set)
    # Each axes now maps to the index of its axes dimension:
    axes_to_dim_mapping = {}
    for axes in axes_set:
        axes_to_dim_mapping[axes] = [axes_dim_list.index(dim) for dim in axes.shape]
    # Each config now maps to the index of its axes:
    config_to_axes_mapping = {}
    for config in config_list:
        config_to_axes_mapping[config] = [
            axes_dim_list.index(dim) for axes in config.axes for dim in axes.shape
        ]
    # Now serialize the axes dimensions, axes, and data configurations:
    serialized_axes_dim_list = []
    for dim in axes_dim_list:
        # Map the operation dimensions to their indices in the axes_dim_list
        if isinstance(dim, OperationDim):
            serialized_axes_dim_list.append(
                {
                    "type": dim.__class__.__name__,
                    "dim_1_index": axes_dim_list.index(dim.dim_1),
                    "dim_2_index": axes_dim_list.index(dim.dim_2),
                }
            )
        else:
            serialized_axes_dim_list.append(
                {
                    "type": dim.__class__.__name__,
                    "size": dim.size,
                    "broadcastable": dim.broadcastable,
                }
            )
    serialized_axes_list = []
    axis_list = []
    for axes in axes_set:
        axis_list.append(axes)
        serialized_axes_list.append(
            {
                "type": axes.__class__.__name__,
                "shape_indices": axes_to_dim_mapping[axes],
            }
        )
    serialized_config_list = []
    for dc in config_list:
        serialized_config_list.append(
            {
                "axes_indices": config_to_axes_mapping[dc],
            }
        )
    return DataConfigSerializationResult(
        {
            KEY_AXES_DIMENSIONS: serialized_axes_dim_list,
            KEY_AXES: serialized_axes_list,
            KEY_DATA_CONFIGURATIONS: serialized_config_list,
        },
        config_list,
        axis_list,
        axes_dim_list,
    )


def order_axes_dims(axes_dim_set: set[AxesDim]) -> list[AxesDim]:
    """
    Orders a set of AxesDim objects based on their dependencies.

    Args:
        axes_dim_set (Set[AxesDim]): A set of AxesDim objects to be ordered.

    Returns:
        list[AxesDim]: A list of AxesDim objects ordered based on their dependencies.
    """
    resolved: set[AxesDim] = set()
    axes_dim_list: list[AxesDim] = []
    remaining = list(axes_dim_set)

    while remaining:
        still_remaining = []
        progressed = False
        for dim in remaining:
            if isinstance(dim, OperationDim) and (
                dim.dim_1 not in resolved or dim.dim_2 not in resolved
            ):
                still_remaining.append(dim)
                continue
            axes_dim_list.append(dim)
            resolved.add(dim)
            progressed = True

        if not progressed:
            raise ValueError(
                f"Could not resolve dependencies for: {still_remaining} "
                "(cyclic or missing dependency)"
            )
        remaining = still_remaining
    return axes_dim_list
