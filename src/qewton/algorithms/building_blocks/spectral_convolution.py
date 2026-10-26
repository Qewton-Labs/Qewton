"""Spectral convolution of Fourier Neural Operators."""

from typing import Annotated, Literal, get_args

from qewton.algorithms.building_blocks.parameters import ParameterNode
from qewton.backends import DEFAULT_DL_BACKEND, Backend, TensorType
from qewton.config.axes import AxesDim, BatchAxes, FeatureAxes, GeometryAxes
from qewton.config.data_configurations import DataConfiguration as DC
from qewton.config.dtypes import Complex64
from qewton.graphs.control_nodes.graph_node import GraphNode
from qewton.graphs.graphs import Graph
from qewton.graphs.nodes import Node

FFTNorm = Literal["forward", "backward", "ortho"]
"""The normalization modes of the fast Fourier transform."""


class FunctionalSpectralConv(Node[TensorType]):
    """Functional spectral convolution of real-valued data.

    The number of modes kept per spatial dimension is inferred from `weight.shape[2:]`.
    Because of the redundancy of the real FFT, the last dimension of the weight has to
    hold `n_modes[-1] // 2 + 1` modes (see `SpectralConv`). If the input has fewer
    frequencies than the weight has modes, the weight is cropped to the available
    ones.

    Attributes:
        input, weight, bias, fft_norm (InputPort): The inputs of the node.
        output (OutputPort): The output of the node.
        batch_axes, geo_axes, feature_axes_x, feature_axes_weight, feature_axes_bias,
            channel_dim_out: Shape information of the data configurations.

    Args:
        n_dim (int): Number of spatial dimensions of the input.
        name (str, optional): Name of the node.
            Defaults to "functional_spectral_conv".
        backend (type[Backend[TensorType]], optional): The backend used for the
            computations. Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        n_dim: int,
        name: str = "functional_spectral_conv",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        # Shape information for the data configurations
        self.batch_axes = BatchAxes(AxesDim(None))
        channel_dim_in, self.channel_dim_out = AxesDim(None), AxesDim(None)
        mode_dims = tuple(AxesDim(None) for _ in range(n_dim))
        self.feature_axes_x = FeatureAxes(shape=(channel_dim_in,))
        self.feature_axes_weight = FeatureAxes(
            shape=(channel_dim_in, self.channel_dim_out, *mode_dims)
        )
        self.feature_axes_bias = FeatureAxes(
            shape=(self.channel_dim_out, *(AxesDim(1) for _ in range(n_dim)))
        )
        self.geo_axes = GeometryAxes(shape=tuple(AxesDim(None) for _ in range(n_dim)))

        super().__init__(name, backend)
        self.input = self.input_ports[0]
        self.weight = self.input_ports[1]
        self.bias = self.input_ports[2]
        self.fft_norm = self.input_ports[3]
        self.output = self.output_ports[0]

    def x_data_config(self) -> DC:
        return DC(self.batch_axes, self.feature_axes_x, self.geo_axes)

    def weight_data_config(self) -> DC:
        return DC(self.feature_axes_weight)

    def bias_data_config(self) -> DC:
        return DC(self.feature_axes_bias)

    def out_data_config(self) -> DC:
        return DC(
            self.batch_axes, FeatureAxes(shape=(self.channel_dim_out,)), self.geo_axes
        )

    def forward(
        self,
        x: Annotated[TensorType, x_data_config],
        weight: Annotated[TensorType, weight_data_config],
        bias: Annotated[TensorType | None, bias_data_config] = None,
        fft_norm: Annotated[FFTNorm | None, DC.empty()] = "forward",
    ) -> Annotated[TensorType, out_data_config]:
        """Applies the spectral convolution.

        Args:
            x (TensorType): Input of shape `(batch, in_channels, *spatial_dims)`.
            weight (TensorType): Complex-valued Fourier weight of shape
                `(in_channels, out_channels, *kept_modes)`, where `kept_modes` already
                has the halving of the last dimension applied.
            bias (TensorType | None, optional): Real-valued bias of shape
                `(out_channels, 1, ..., 1)`, broadcast over batch and space.
                Defaults to None.
            fft_norm (FFTNorm | None, optional): Normalization of the forward and
                the inverse FFT. Defaults to "forward".

        Returns:
            TensorType: Output of shape `(batch, out_channels, *spatial_dims)`.
        """
        self._check_input(x, weight)
        math = self.backend.math
        n_spatial_dims = weight.ndim - 2
        fft_dims = list(range(2, 2 + n_spatial_dims))
        shifted_dims = fft_dims[:-1]

        spatial_shape = x.shape[2:]
        spectrum_shape = list(spatial_shape)
        spectrum_shape[-1] = spectrum_shape[-1] // 2 + 1  # real FFT: half spectrum

        x_spectrum = math.rfftn(x, dim=fft_dims, norm=fft_norm)
        # Center the zero frequency along all but the last dimension, so that the
        # modes to keep can be selected around it.
        if n_spatial_dims > 1:
            x_spectrum = math.fftshift(x_spectrum, dim=shifted_dims)

        weight = self._crop_weight(weight, spectrum_shape)
        mode_slices = self._mode_slices(spectrum_shape, weight.shape[2:])
        out_spectrum = math.zeros(
            (x.shape[0], weight.shape[1], *spectrum_shape),
            dtype=x_spectrum.dtype,
            device=x_spectrum.device,
        )
        out_spectrum[mode_slices] = math.einsum(
            "bi...,io...->bo...", x_spectrum[mode_slices], weight
        )

        if n_spatial_dims > 1:
            out_spectrum = math.ifftshift(out_spectrum, dim=shifted_dims)
        out = math.irfftn(out_spectrum, s=spatial_shape, dim=fft_dims, norm=fft_norm)
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def _check_input(x: TensorType, weight: TensorType) -> None:
        """Checks that the input fits to the weight.

        Args:
            x (TensorType): Input of shape `(batch, in_channels, *spatial_dims)`.
            weight (TensorType): Weight of shape `(in_channels, out_channels, *modes)`.

        Raises:
            ValueError: If the input has a different number of dimensions or input
                channels than the weight.
        """
        if x.ndim != weight.ndim or x.shape[1] != weight.shape[0]:
            spatial_dims = ", ".join(["x"] * (weight.ndim - 2))
            raise ValueError(
                f"Expected an input of shape (batch, {weight.shape[0]}, "
                f"{spatial_dims}), got {tuple(x.shape)}."
            )

    @staticmethod
    def _crop_weight(weight: TensorType, spectrum_shape: list[int]) -> TensorType:
        """Removes the modes of the weight that do not fit into the spectrum.

        This happens for inputs with fewer frequencies than the weight has modes. In
        all but the last dimension the modes are removed from both ends (one more from
        the high end if the number is odd), in the last dimension the highest ones.

        Args:
            weight (TensorType): Weight of shape `(in_channels, out_channels, *modes)`.
            spectrum_shape (list[int]): Shape of the spectrum of the input, with the
                half spectrum in the last dimension.

        Returns:
            TensorType: The weight with at most `spectrum_shape` modes per dimension.
        """
        n_removed = [
            max(n_modes - size, 0)
            for size, n_modes in zip(spectrum_shape, weight.shape[2:])
        ]
        slices = [slice(None), slice(None)]  # input and output channels
        for n in n_removed[:-1]:
            n_removed_low = n // 2
            n_removed_high = n - n_removed_low
            slices.append(slice(n_removed_low, -n_removed_high if n else None))
        slices.append(slice(None, -n_removed[-1] if n_removed[-1] else None))
        return weight[tuple(slices)]

    @staticmethod
    def _mode_slices(
        spectrum_shape: list[int], kept_modes: tuple[int, ...]
    ) -> tuple[slice, ...]:
        """Selects the kept modes of the (shifted) spectrum.

        Args:
            spectrum_shape (list[int]): Shape of the spectrum of the input, with the
                half spectrum in the last dimension.
            kept_modes (tuple[int, ...]): Number of modes to keep in every dimension.

        Returns:
            tuple[slice, ...]: Slices for `(batch, channels, *spectrum_dims)`: a window
                around the zero frequency in all but the last dimension, and the lowest
                frequencies in the last one.
        """
        slices = [slice(None), slice(None)]  # batch and channels
        for size, n_kept in zip(spectrum_shape[:-1], kept_modes[:-1]):
            center = size // 2
            n_negative = n_kept // 2
            n_positive = n_kept - n_negative
            slices.append(slice(center - n_negative, center + n_positive))
        slices.append(slice(0, kept_modes[-1]))
        return tuple(slices)


class SpectralConv(GraphNode[TensorType]):
    """Spectral convolution layer of a Fourier Neural Operator.

    Owns a complex-valued Fourier weight (with the mode count of the last spatial
    dimension halved because of the redundancy of the real FFT) and an optional
    real-valued bias per output channel. The computation is delegated to
    `FunctionalSpectralConv`.

    Attributes:
        kept_modes (tuple[int, ...]): Shape of the weight along the spatial
            dimensions, i.e. `n_modes` with the last entry halved.
        weight (ParameterNode): The complex Fourier weight of shape
            `(in_channels, out_channels, *kept_modes)`.
        bias (ParameterNode | None): The bias of shape `(out_channels, 1, ..., 1)`,
            or None if the layer has no bias.

    Raises:
        ValueError: If a number of channels or modes is not positive, `init_std` is
            negative, or `fft_norm` is unknown.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_modes (tuple[int, ...]): Number of modes to keep in every spatial dimension.
        bias (bool, optional): If a bias should be added. Defaults to True.
        init_std (float | Literal["auto"], optional): Standard deviation of the
            normal distribution used to initialize weight and bias. "auto" uses
            `sqrt(2 / (in_channels + out_channels))`. Defaults to "auto".
        fft_norm (FFTNorm | None, optional): Normalization of the forward and the
            inverse FFT. Defaults to "forward".
        name (str, optional): Name of the node. Defaults to "spectral_conv".
        backend (type[Backend[TensorType]], optional): The backend used for the
            computations. Defaults to DEFAULT_DL_BACKEND.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        bias: bool = True,
        init_std: float | Literal["auto"] = "auto",
        fft_norm: FFTNorm | None = "forward",
        name: str = "spectral_conv",
        backend: type[Backend[TensorType]] = DEFAULT_DL_BACKEND,
    ) -> None:
        self._check_arguments(in_channels, out_channels, n_modes, init_std, fft_norm)
        self.fft_norm = fft_norm
        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5

        self.n_modes = n_modes
        self.kept_modes = (*n_modes[:-1], n_modes[-1] // 2 + 1)

        weight_shape = (in_channels, out_channels, *self.kept_modes)
        self.weight = ParameterNode(
            weight_shape,
            initial_value=backend.random.normal(
                weight_shape, std=init_std, dtype=backend.dtypes[Complex64]
            ),
            dtype=backend.dtypes[Complex64],
            name="weight",
            backend=backend,
        )
        if bias:
            bias_shape = (out_channels, *(1,) * len(n_modes))
            self.bias = ParameterNode(
                bias_shape,
                initial_value=backend.random.normal(bias_shape, std=init_std),
                name="bias",
                backend=backend,
            )
        else:
            self.bias = None

        self.functional_spectral_conv = FunctionalSpectralConv(
            len(n_modes), backend=backend
        )

        graph = Graph()
        graph.connect(self.weight, self.functional_spectral_conv.weight)
        if bias:
            graph.connect(self.bias, self.functional_spectral_conv.bias)

        super().__init__(
            graph=graph,
            input_ports=[self.functional_spectral_conv.input],
            output_ports=[self.functional_spectral_conv.output],
            name=name,
            backend=backend,
        )
        self.functional_spectral_conv.fft_norm.default = self.fft_norm
        self._graph.setup()
        self.input = self.input_ports[0]
        self.output = self.output_ports[0]

    @staticmethod
    def _check_arguments(
        in_channels: int,
        out_channels: int,
        n_modes: tuple[int, ...],
        init_std: float | Literal["auto"],
        fft_norm: FFTNorm | None,
    ) -> None:
        """Checks the arguments of `SpectralConv`.

        Raises:
            ValueError: If a number of channels or modes is not positive, `init_std` is
                negative, or `fft_norm` is unknown.
        """
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                "in_channels and out_channels must be positive, "
                f"got {in_channels} and {out_channels}."
            )
        if len(n_modes) == 0 or min(n_modes) < 1:
            raise ValueError(f"n_modes must contain positive integers, got {n_modes}.")
        if init_std != "auto" and init_std < 0:
            raise ValueError(f"init_std must be 'auto' or not negative, got {init_std}.")
        if fft_norm is not None and fft_norm not in get_args(FFTNorm):
            raise ValueError(
                f"fft_norm must be one of {get_args(FFTNorm)}, got {fft_norm!r}."
            )

    def forward(self, x: TensorType) -> TensorType:
        """Applies the spectral convolution.

        Args:
            x (TensorType): Input of shape `(batch, in_channels, *spatial_dims)`.

        Returns:
            TensorType: Output of shape `(batch, out_channels, *spatial_dims)`.
        """
        self.input.set_value(x)
        self.run()
        return self.output.value
