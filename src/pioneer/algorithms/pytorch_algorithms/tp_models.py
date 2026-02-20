import torch
import torchphysics as tp

from ...config.configuration_base import DataConfiguration
from ...config.variables import Variable
from ...config.axis import BatchAxis, FeatureAxis, SpatialAxis
from .wrapper import PyTorchWrapper
from ..base import AlgorithmAttributes, AlgorithmState


class TorchPhysicsFNO(PyTorchWrapper):

    def __init__(
        self,
        input_variable: Variable,
        output_variable: Variable,
        spatial_dimension: int,
        name: str = "FNO",
        **kwargs,
    ) -> None:
        input_feature_axis = FeatureAxis(variables=input_variable)
        output_feature_axis = FeatureAxis(variables=output_variable)
        spatial_axis = [SpatialAxis()] * spatial_dimension
        input_config = DataConfiguration(
            dtype=torch.float,
            axes=[BatchAxis(), *spatial_axis, input_feature_axis],
            feature_axis=input_feature_axis,
        )
        output_config = DataConfiguration(
            dtype=torch.float,
            axes=[BatchAxis(), *spatial_axis, output_feature_axis],
            feature_axis=output_feature_axis,
        )
        self.input_space = tp.spaces.Space(input_variable)
        kwargs["input_space"] = self.input_space
        kwargs["output_space"] = tp.spaces.Space(output_variable)
        super().__init__(
            input_config, output_config, tp.models.FNO, name, **kwargs  # type: ignore
        )
        self.set_attributes(
            AlgorithmAttributes.DIFFERENTIABLE, AlgorithmAttributes.TRAINABLE
        )

    def run(
        self, inputs: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        if self.state == AlgorithmState.UNINITIALIZED:
            self.setup()
        data = inputs[self.InputKeys.INPUT]  # type: ignore
        outdata = self.model(tp.spaces.Points(data, self.input_space))
        return {self.OutputKeys.OUTPUT: outdata.as_tensor}
