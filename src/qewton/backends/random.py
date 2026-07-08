from typing import Any, Optional
from qewton.backends.base import Backend, TensorType
from qewton.config.devices import Device, cpu


class RandomBackend(Backend[TensorType]):
    """A Backend that implements random number generation.
    Method selection and behavior inspired by numpy.random to unify
    usage across different backends.
    """

    @staticmethod
    def set_seed(seed: int):
        """Sets the seed for the random number generator."""
        raise NotImplementedError

    @staticmethod
    def normal(
        shape: Any,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a normal (Gaussian) distribution."""
        raise NotImplementedError

    @staticmethod
    def standard_normal(
        shape: Any, dtype: Any = None, device: Device | str = cpu
    ) -> TensorType:
        """Samples from a standard normal distribution (mean=0, std=1)."""
        raise NotImplementedError

    @staticmethod
    def uniform(
        shape: Any,
        low: float = 0.0,
        high: float = 1.0,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a uniform distribution over [low, high)."""
        raise NotImplementedError

    @staticmethod
    def randint(
        low: int,
        high: Optional[int] = None,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a discrete uniform distribution over [low, high)."""
        raise NotImplementedError

    @staticmethod
    def choice(
        a: Any,
        shape: Any = None,
        replace: bool = True,
        p: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Generates a random sample from a given 1-D array or integer."""
        raise NotImplementedError

    @staticmethod
    def permutation(x: Any, device: Device | str = cpu) -> TensorType:
        """Randomly permute a sequence, or return a permuted range."""
        raise NotImplementedError

    @staticmethod
    def exponential(
        shape: Any, scale: float = 1.0, dtype: Any = None, device: Device | str = cpu
    ) -> TensorType:
        """Samples from an exponential distribution."""
        raise NotImplementedError

    @staticmethod
    def multivariate_normal(
        mean: Any,
        cov: Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a multivariate normal distribution."""
        raise NotImplementedError

    @staticmethod
    def shuffle(x: Any, axis: int = 0) -> None:
        """Modify a sequence in-place by shuffling its contents."""
        raise NotImplementedError

    @staticmethod
    def binomial(
        n: int | Any,
        p: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a binomial distribution."""
        raise NotImplementedError

    @staticmethod
    def poisson(
        lam: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a Poisson distribution."""
        raise NotImplementedError

    @staticmethod
    def gamma(
        shape_param: float | Any,
        scale: float | Any = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a Gamma distribution."""
        raise NotImplementedError

    @staticmethod
    def beta(
        a: float | Any,
        b: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a Beta distribution."""
        raise NotImplementedError

    @staticmethod
    def lognormal(
        mean: float = 0.0,
        sigma: float = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a log-normal distribution."""
        raise NotImplementedError

    @staticmethod
    def gumbel(
        loc: float = 0.0,
        scale: float = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> TensorType:
        """Samples from a Gumbel distribution."""
        raise NotImplementedError
