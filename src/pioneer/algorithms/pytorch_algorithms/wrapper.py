import inspect
from typing import get_type_hints, get_origin
import torch

from ...config.configuration_base import DataConfiguration
from ..base import AlgorithmAttributes
from ...nodes.base import Port
from ...optim.hyperparameter.number_hyperparameter import (
    HyperParameter,
    DiscreteHyperparameter,
    ContinuousHyperparameter,
)
from ...optim.hyperparameter.categorical_hyperparameter import (
    CategoricalHyperparameter,
    BooleanHyperparameter,
)


class PyTorchWrapper(AlgorithmNode):
    """Implements a wrapper for all torch.nn.Modules to use them in the
    pipeline structure of this library.
    """

    def __init__(
        self,
        input_config: DataConfiguration,
        output_config: DataConfiguration,
        model_cls: type[torch.nn.Module],
        name: str = "PyTorchAlgorithm",
        **kwargs,
    ) -> None:
        """
        Args:
            input_config (DataConfiguration): The expected input shape for the
                models that are wrapped.
            output_config (DataConfiguration): The expected output shape for the
                models that are wrapped.
            model_cls (type[torch.nn.Module]): The class object that is a subclass
                of torch.nn.Module.
            name (str, optional): The name of this node.
                Defaults to "PyTorchAlgorithm".
            **kwargs (Any): The input arguments of the class provided in *model_cls*.
                Can either be the values needed to construct an object from the
                class or corresponding HyperParameters for tuning.
        """
        input_variable = input_config.feature_axis.variables  # type: ignore
        output_variable = output_config.feature_axis.variables  # type: ignore
        super().__init__(input_variable, output_variable, name)  # type: ignore

        self.input_port = Port(input_config, self, "InputPort", True)
        self.output_port = Port(output_config, self, "OutputPort")

        self.model: torch.nn.Module = torch.nn.Module()
        self.model_cls = model_cls
        self.kwargs: dict[str, HyperParameter] = {}
        self._check_input_args(**kwargs)
        self._attributes = set[AlgorithmAttributes]()

    def _check_input_args(self, **kwargs):
        """Checks if the provided kwargs with the expected input of self.model_cls.

        Raises:
            TypeError: If a kwarg does not fit or is missing from the __init__
                call of self.model_cls.
            TypeError: If a kwarg has a wrong type.
        """
        hyperparameter_map = {
            int: (DiscreteHyperparameter,),
            float: (ContinuousHyperparameter,),
            bool: (BooleanHyperparameter,),
        }

        # Check if kwargs fit the model_cls
        try:
            cls_sig = inspect.signature(self.model_cls)
            cls_sig.bind(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Invalid arguments for {self.model_cls.__name__}: {e}"
            ) from e

        type_hints = get_type_hints(self.model_cls.__init__)
        for kw, val in kwargs.items():
            expected_type = type_hints.get(kw)
            if expected_type is not None:
                # Get the runtime-checkable origin
                origin = get_origin(expected_type) or expected_type
                allowed_types = (origin,)
                # Accept either the real type or corresponding hyperparameter
                if expected_type in hyperparameter_map:
                    allowed_types += hyperparameter_map[expected_type]
                else:
                    allowed_types += (CategoricalHyperparameter,)

                if not isinstance(val, allowed_types):
                    allowed_names = ", ".join([t.__name__ for t in allowed_types])
                    raise TypeError(
                        f"Argument '{kw}' has wrong type: expected {allowed_names}, "
                        f"got {type(val).__name__}"
                    )

            # Transform any argument to Hyperparameters for a consistent use
            # internally.
            self.kwargs[kw] = HyperParameter.from_value(val, name=kw)

    def setup(self) -> None:
        current_kwargs = {}
        for key, value in self.kwargs.items():
            current_kwargs[key] = value.value
        self.model = self.model_cls(**current_kwargs)
        self._state = AlgorithmState.READY

    def _run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.state == AlgorithmState.UNINITIALIZED:
            self.setup()
        data = inputs[self.InputKeys.INPUT]  # type: ignore
        outdata = self.model(data)
        return {self.OutputKeys.OUTPUT: outdata}

    @property
    def input_ports(self) -> dict[str, Port]:
        return {self.InputKeys.INPUT: self.input_port}

    @property
    def output_ports(self) -> dict[str, Port]:
        return {self.OutputKeys.OUTPUT: self.output_port}

    @property
    def trainable_parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model = self.model.to(device=device)

    def reset(self):
        if not self.state == AlgorithmState.FIXED:
            self.model: torch.nn.Module = torch.nn.Module()
            self._state = AlgorithmState.UNINITIALIZED

    @property
    def hyperparameters(self) -> list[HyperParameter]:
        return list(self.kwargs.values())

    @property
    def attributes(self) -> set[AlgorithmAttributes]:
        return self._attributes

    def set_attributes(self, *attributes: AlgorithmAttributes):
        for attribute in attributes:
            self._attributes.add(attribute)
