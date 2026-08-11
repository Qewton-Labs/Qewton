from typing import Any, Optional
import torch

from qewton.backends.random import RandomBackend
from qewton.backends.torch.device import get_torch_device
from qewton.config.devices import Device, cpu


class TorchRandomBackend(RandomBackend[torch.Tensor]):

    @staticmethod
    def set_seed(seed: int):
        """Sets the seed for the random number generator."""
        torch.random.manual_seed(seed)

    @staticmethod
    def normal(
        shape: Any,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a normal (Gaussian) distribution."""
        return torch.normal(
            mean=mean,
            std=std,
            size=shape,
            device=get_torch_device(device),
            dtype=dtype,
        )

    @staticmethod
    def standard_normal(
        shape: Any, dtype: Any = None, device: Device | str = cpu
    ) -> torch.Tensor:
        """Samples from a standard normal distribution (mean=0, std=1)."""
        return torch.randn(shape, device=get_torch_device(device), dtype=dtype)

    @staticmethod
    def uniform(
        shape: Any,
        low: float = 0.0,
        high: float = 1.0,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a uniform distribution over [low, high)."""
        return low + (high - low) * torch.rand(
            shape, device=get_torch_device(device), dtype=dtype
        )

    @staticmethod
    def randint(
        low: int,
        high: Optional[int] = None,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a discrete uniform distribution over [low, high)."""
        if high is None:
            high = low
            low = 0
        if shape is None:
            shape = ()
        return torch.randint(
            low, high, size=shape, device=get_torch_device(device), dtype=dtype
        )

    @staticmethod
    def choice(
        a: Any,
        shape: Any = None,
        replace: bool = True,
        p: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Generates a random sample from a given 1-D array or integer."""
        torch_device = get_torch_device(device)
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(a, int):
            a_tensor = torch.arange(a, device=torch_device)
        else:
            a_tensor = torch.as_tensor(a, device=torch_device)

        num_elements = a_tensor.size(0)
        num_samples = 1 if shape is None else torch.Size(shape).numel()

        if p is not None:
            p_tensor = torch.as_tensor(p, device=torch_device)
            indices = torch.multinomial(p_tensor, num_samples, replacement=replace)
        else:
            if replace:
                indices = torch.randint(
                    0, num_elements, (num_samples,), device=torch_device
                )
            else:
                indices = torch.randperm(num_elements, device=torch_device)[:num_samples]

        samples = a_tensor[indices]
        if shape is not None:
            samples = samples.reshape(shape)
        return samples

    @staticmethod
    def permutation(x: Any, device: Device | str = cpu) -> torch.Tensor:
        """Randomly permute a sequence, or return a permuted range."""
        torch_device = get_torch_device(device)
        if isinstance(x, int):
            return torch.randperm(x, device=torch_device)

        x_tensor = torch.as_tensor(x, device=torch_device)
        return x_tensor[torch.randperm(x_tensor.size(0), device=torch_device)]

    @staticmethod
    def exponential(
        shape: Any, scale: float = 1.0, dtype: Any = None, device: Device | str = cpu
    ) -> torch.Tensor:
        """Samples from an exponential distribution."""
        return torch.empty(
            shape, device=get_torch_device(device), dtype=dtype
        ).exponential_(1.0 / scale)

    @staticmethod
    def multivariate_normal(
        mean: Any,
        cov: Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a multivariate normal distribution."""
        torch_device = get_torch_device(device)
        mean_t = torch.as_tensor(mean, device=torch_device, dtype=dtype)
        cov_t = torch.as_tensor(cov, device=torch_device, dtype=dtype)
        dist = torch.distributions.MultivariateNormal(mean_t, cov_t)
        if shape is None:
            return dist.sample()
        if isinstance(shape, int):
            shape = (shape,)
        return dist.sample(torch.Size(shape))

    @staticmethod
    def shuffle(x: Any, axis: int = 0) -> None:
        """Modify a sequence in-place by shuffling its contents."""
        indices = torch.randperm(x.size(axis), device=x.device)
        shuffled = x.index_select(axis, indices)
        x.copy_(shuffled)

    @staticmethod
    def binomial(
        n: int | Any,
        p: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a binomial distribution."""
        torch_device = get_torch_device(device)
        n_t = torch.as_tensor(n, device=torch_device)
        p_t = torch.as_tensor(p, device=torch_device)
        dist = torch.distributions.Binomial(total_count=n_t, probs=p_t)
        res = dist.sample(torch.Size(shape) if shape is not None else ())
        if dtype is not None:
            res = res.to(dtype)
        return res

    @staticmethod
    def poisson(
        lam: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a Poisson distribution."""
        torch_device = get_torch_device(device)
        rate = torch.full(shape if shape is not None else (), lam, device=torch_device)
        res = torch.poisson(rate)
        if dtype is not None:
            res = res.to(dtype)
        return res

    @staticmethod
    def gamma(
        shape_param: float | Any,
        scale: float | Any = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a Gamma distribution."""
        torch_device = get_torch_device(device)
        dist = torch.distributions.Gamma(
            concentration=torch.as_tensor(shape_param, device=torch_device),
            rate=torch.as_tensor(1.0 / scale, device=torch_device),
        )
        res = dist.sample(torch.Size(shape) if shape is not None else ())
        if dtype is not None:
            res = res.to(dtype)
        return res

    @staticmethod
    def beta(
        a: float | Any,
        b: float | Any,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a Beta distribution."""
        torch_device = get_torch_device(device)
        dist = torch.distributions.Beta(
            torch.as_tensor(a, device=torch_device),
            torch.as_tensor(b, device=torch_device),
        )
        res = dist.sample(torch.Size(shape) if shape is not None else ())
        if dtype is not None:
            res = res.to(dtype)
        return res

    @staticmethod
    def lognormal(
        mean: float = 0.0,
        sigma: float = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a log-normal distribution."""
        torch_device = get_torch_device(device)
        return torch.empty(
            shape if shape is not None else (), device=torch_device, dtype=dtype
        ).log_normal_(mean, sigma)

    @staticmethod
    def gumbel(
        loc: float = 0.0,
        scale: float = 1.0,
        shape: Any = None,
        dtype: Any = None,
        device: Device | str = cpu,
    ) -> torch.Tensor:
        """Samples from a Gumbel distribution."""
        torch_device = get_torch_device(device)
        dist = torch.distributions.Gumbel(
            torch.as_tensor(loc, device=torch_device),
            torch.as_tensor(scale, device=torch_device),
        )
        res = dist.sample(torch.Size(shape) if shape is not None else ())
        if dtype is not None:
            res = res.to(dtype)
        return res
