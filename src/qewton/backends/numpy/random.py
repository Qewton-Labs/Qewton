from typing import Any, Optional
import numpy as np

from qewton.backends.random import RandomBackend
from qewton.config.devices import Device, cpu


class NumpyRandomBackend(RandomBackend[np.ndarray]):

    @staticmethod
    def set_seed(seed: int):
        np.random.seed(seed)

    @staticmethod
    def normal(
        shape: Any,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        """Samples from a normal (Gaussian) distribution."""
        result = np.random.normal(loc=mean, scale=std, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def standard_normal(
        shape: Any, dtype: Any = None, device: Device = cpu
    ) -> np.ndarray:
        """Samples from a standard normal distribution (mean=0, std=1)."""
        result = np.random.standard_normal(size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def uniform(
        shape: Any,
        low: float = 0.0,
        high: float = 1.0,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        """Samples from a uniform distribution over [low, high)."""
        result = np.random.uniform(low=low, high=high, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def randint(
        low: int,
        high: Optional[int] = None,
        shape: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        if dtype is None:
            dtype = np.int32
        result = np.random.randint(low, high=high, size=shape, dtype=dtype)
        return result

    @staticmethod
    def choice(
        a: Any,
        shape: Any = None,
        replace: bool = True,
        p: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        result = np.random.choice(a, size=shape, replace=replace, p=p)
        return result

    @staticmethod
    def permutation(x: Any, device: Device = cpu) -> np.ndarray:
        """Randomly permute a sequence, or return a permuted range."""
        return np.random.permutation(x)

    @staticmethod
    def exponential(
        shape: Any, scale: float = 1.0, dtype: Any = None, device: Device = cpu
    ) -> np.ndarray:
        """Samples from an exponential distribution."""
        result = np.random.exponential(scale=scale, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def multivariate_normal(
        mean: Any, cov: Any, shape: Any = None, dtype: Any = None, device: Device = cpu
    ) -> np.ndarray:
        """Samples from a multivariate normal distribution."""
        result = np.random.multivariate_normal(mean=mean, cov=cov, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def shuffle(x: Any, axis: int = 0) -> None:
        """Modify a sequence in-place by shuffling its contents."""
        # NumPy's shuffle operates in-place and does not have an 'axis' argument.
        # The 'axis' parameter in the signature will be ignored for NumPy.
        np.random.shuffle(x)

    @staticmethod
    def binomial(
        n: int | Any,
        p: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        """Samples from a binomial distribution."""
        result = np.random.binomial(n=n, p=p, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def poisson(
        lam: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        """Samples from a Poisson distribution."""
        result = np.random.poisson(lam=lam, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def gamma(
        shape_param: float | Any,
        scale: float | Any = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        """Samples from a Gamma distribution."""
        # Note: np.random.gamma uses 'shape' for the distribution's shape parameter.
        result = np.random.gamma(shape=shape_param, scale=scale, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def beta(
        a: float | Any,
        b: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        """Samples from a Beta distribution."""
        result = np.random.beta(a=a, b=b, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def lognormal(
        mean: float = 0.0,
        sigma: float = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        """Samples from a log-normal distribution."""
        result = np.random.lognormal(mean=mean, sigma=sigma, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result

    @staticmethod
    def gumbel(
        loc: float = 0.0,
        scale: float = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device = cpu,
    ) -> np.ndarray:
        """Samples from a Gumbel distribution."""
        result = np.random.gumbel(loc=loc, scale=scale, size=shape)
        if dtype is not None:
            return result.astype(dtype)
        return result
