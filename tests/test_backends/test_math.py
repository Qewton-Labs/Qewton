import inspect
import math
import pytest

from qewton.backends.base import Backend
from qewton.backends.torch.base import TorchBackend


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "math"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = [TorchBackend]  # all_subclasses(Backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_all(backend):
    in_1 = backend.build_tensor([[True, True], [True, True]])
    assert backend.math.all(in_1)
    in_1 = backend.build_tensor([True, True, False, True])
    assert not backend.math.all(in_1)
    in_1 = backend.build_tensor([])
    assert backend.math.all(in_1)  # Empty array should return True


@pytest.mark.parametrize("backend", BACKENDS)
def test_add(backend):
    in_1 = backend.build_tensor([[1, 0.0]])
    in_2 = backend.build_tensor([[2.0, -10.0]])
    out = backend.build_tensor([[3.0, -10.0]])
    assert backend.math.all(backend.math.add(in_1, in_2) == out)

    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor(10)
    out = backend.build_tensor([11, 12, 13])
    assert backend.math.all(backend.math.add(in_1, in_2) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    in_2 = backend.build_tensor([[10, 20], [30, 40]])
    out = backend.build_tensor([[11, 22], [33, 44]])
    assert backend.math.all(backend.math.add(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_multiply(backend):
    in_1 = backend.build_tensor([[1, 2.0]])
    in_2 = backend.build_tensor([[2.0, -10.0]])
    out = backend.build_tensor([[2.0, -20.0]])
    assert backend.math.all(backend.math.multiply(in_1, in_2) == out)

    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor(2)
    out = backend.build_tensor([2, 4, 6])
    assert backend.math.all(backend.math.multiply(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_subtract(backend):
    in_1 = backend.build_tensor([[3, 0.0]])
    in_2 = backend.build_tensor([[2.0, -10.0]])
    out = backend.build_tensor([[1.0, 10.0]])
    assert backend.math.all(backend.math.subtract(in_1, in_2) == out)

    in_1 = backend.build_tensor([10, 20, 30])
    in_2 = backend.build_tensor(5)
    out = backend.build_tensor([5, 15, 25])
    assert backend.math.all(backend.math.subtract(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_matmul(backend):
    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    in_2 = backend.build_tensor([[5, 6], [7, 8]])
    out = backend.build_tensor([[19, 22], [43, 50]])
    assert backend.math.all(backend.math.matmul(in_1, in_2) == out)

    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([[3], [4]])
    out = backend.build_tensor([11])
    assert backend.math.all(backend.math.matmul(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mean(backend):
    in_1 = backend.build_tensor([[1.0, 2.0], [3.0, 4.0]])
    out = backend.build_tensor(2.5)
    assert backend.math.all(backend.math.mean(in_1) == out)

    in_1 = backend.build_tensor([[1.0, 2.0], [3.0, 4.0]])
    out = backend.build_tensor([2.0, 3.0])
    assert backend.math.all(backend.math.mean(in_1, axis=0) == out)

    in_1 = backend.build_tensor([[1.0, 2.0], [3.0, 4.0]])
    out = backend.build_tensor([[2.0], [3.0]])
    assert backend.math.all(backend.math.mean(in_1, axis=1, keepdims=True) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_max(backend):
    in_1 = backend.build_tensor([[1, 5], [3, 2]])
    out = backend.build_tensor(5)
    assert backend.math.all(backend.math.max(in_1) == out)

    in_1 = backend.build_tensor([[1, 5], [3, 2]])
    out = backend.build_tensor([3, 5])
    assert backend.math.all(backend.math.max(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_min(backend):
    in_1 = backend.build_tensor([[1, 5], [3, 2]])
    out = backend.build_tensor(1)
    assert backend.math.all(backend.math.min(in_1) == out)

    in_1 = backend.build_tensor([[1, 5], [3, 2]])
    out = backend.build_tensor([1, 2])
    assert backend.math.all(backend.math.min(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ones(backend):
    out = backend.build_tensor([[1.0, 1.0], [1.0, 1.0]])
    assert backend.math.all(backend.math.ones((2, 2)) == out)

    out = backend.build_tensor([1, 1, 1])
    assert backend.math.all(backend.math.ones(3) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zeros(backend):
    out = backend.build_tensor([[0.0, 0.0], [0.0, 0.0]])
    assert backend.math.all(backend.math.zeros((2, 2)) == out)

    out = backend.build_tensor([0, 0, 0])
    assert backend.math.all(backend.math.zeros(3) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_eye(backend):
    out = backend.build_tensor([[1.0, 0.0], [0.0, 1.0]])
    assert backend.math.all(backend.math.eye(2) == out)

    out = backend.build_tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert backend.math.all(backend.math.eye(2, M=3, k=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_linspace(backend):
    out = backend.build_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert backend.math.allclose(backend.math.linspace(1, 5, num=5), out)

    out = backend.build_tensor([1.0, 1.5, 2.0, 2.5, 3.0])
    assert backend.math.allclose(backend.math.linspace(1, 3, num=5), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_arange(backend):
    out = backend.build_tensor([0, 1, 2, 3, 4])
    assert backend.math.all(backend.math.arange(5) == out)

    out = backend.build_tensor([2, 4, 6, 8])
    assert backend.math.all(backend.math.arange(2, 10, 2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_absolute(backend):
    in_1 = backend.build_tensor([-1, 0, 1, -2.5])
    out = backend.build_tensor([1, 0, 1, 2.5])
    assert backend.math.all(backend.math.absolute(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_any(backend):
    in_1 = backend.build_tensor([[True, False], [False, False]])
    assert backend.math.any(in_1)

    in_1 = backend.build_tensor([False, False, False])
    assert not backend.math.any(in_1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_append(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor([4, 5])
    out = backend.build_tensor([1, 2, 3, 4, 5])
    assert backend.math.all(backend.math.append(in_1, in_2) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    in_2 = backend.build_tensor([[5, 6]])
    out = backend.build_tensor([[1, 2], [3, 4], [5, 6]])
    assert backend.math.all(backend.math.append(in_1, in_2, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_arccos(backend):
    in_1 = backend.build_tensor([1.0, 0.0, -1.0])
    out = backend.build_tensor([0.0, math.pi / 2, math.pi])
    assert backend.math.allclose(backend.math.arccos(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_arctan2(backend):
    in_1 = backend.build_tensor([1, 1, -1, -1])
    in_2 = backend.build_tensor([1, -1, -1, 1])
    out = backend.build_tensor(
        [math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4, -math.pi / 4]
    )
    assert backend.math.allclose(backend.math.arctan2(in_1, in_2), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_argmax(backend):
    in_1 = backend.build_tensor([1, 5, 2, 8, 3])
    out = backend.build_tensor(3)
    assert backend.math.all(backend.math.argmax(in_1) == out)

    in_1 = backend.build_tensor([[1, 5, 2], [8, 3, 4]])
    out = backend.build_tensor([1, 0])
    assert backend.math.all(backend.math.argmax(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_argmin(backend):
    in_1 = backend.build_tensor([1, 5, 2, 8, 3])
    out = backend.build_tensor(0)
    assert backend.math.all(backend.math.argmin(in_1) == out)

    in_1 = backend.build_tensor([[1, 5, 2], [8, 3, 4]])
    out = backend.build_tensor([0, 1])
    assert backend.math.all(backend.math.argmin(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_argsort(backend):
    in_1 = backend.build_tensor([3, 1, 4, 1, 5, 9, 2, 6])
    out = backend.build_tensor([1, 3, 6, 0, 2, 7, 4, 5])
    assert backend.math.all(backend.math.argsort(in_1) == out)

    in_1 = backend.build_tensor([[0, 3], [2, 1]])
    out = backend.build_tensor([[0, 1], [1, 0]])
    assert backend.math.all(backend.math.argsort(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_array(backend):
    data = [[1, 2], [3, 4]]
    out = backend.build_tensor(data)
    assert backend.math.all(backend.math.array(data) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_broadcast_to(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    shape = (2, 3)
    out = backend.build_tensor([[1, 2, 3], [1, 2, 3]])
    assert backend.math.all(backend.math.broadcast_to(in_1, shape) == out)

    in_1 = backend.build_tensor(5)
    shape = (2, 2)
    out = backend.build_tensor([[5, 5], [5, 5]])
    assert backend.math.all(backend.math.broadcast_to(in_1, shape) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ceil(backend):
    in_1 = backend.build_tensor([1.1, 2.9, -3.1, 0.0])
    out = backend.build_tensor([2.0, 3.0, -3.0, 0.0])
    assert backend.math.all(backend.math.ceil(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_clip(backend):
    in_1 = backend.build_tensor([0, 1, 2, 3, 4, 5])
    out = backend.build_tensor([1, 1, 2, 3, 3, 3])
    assert backend.math.all(backend.math.clip(in_1, 1, 3) == out)

    in_1 = backend.build_tensor([-10, 0, 10])
    out = backend.build_tensor([0, 0, 10])
    assert backend.math.all(backend.math.clip(in_1, 0, None) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_concatenate(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor([1, 2, 3, 4])
    assert backend.math.all(backend.math.concatenate([in_1, in_2]) == out)

    in_1 = backend.build_tensor([[1, 2]])
    in_2 = backend.build_tensor([[3, 4]])
    out = backend.build_tensor([[1, 2], [3, 4]])
    assert backend.math.all(backend.math.concatenate([in_1, in_2], axis=0) == out)

    in_1 = backend.build_tensor([[1], [2]])
    in_2 = backend.build_tensor([[3], [4]])
    out = backend.build_tensor([[1, 3], [2, 4]])
    assert backend.math.all(backend.math.concatenate([in_1, in_2], axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_copy(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    copied = backend.math.copy(in_1)
    assert backend.math.all(copied == in_1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cos(backend):
    in_1 = backend.build_tensor([0.0, math.pi / 2, math.pi])
    out = backend.build_tensor([1.0, 0.0, -1.0])
    assert backend.math.allclose(backend.math.cos(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cosh(backend):
    in_1 = backend.build_tensor([0.0, 1.0])
    out = backend.build_tensor([math.cosh(0.0), math.cosh(1.0)])
    assert backend.math.allclose(backend.math.cosh(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_count_nonzero(backend):
    in_1 = backend.build_tensor([0, 1, 0, 2, 0, 3])
    out = backend.build_tensor(3)
    assert backend.math.all(backend.math.count_nonzero(in_1) == out)

    in_1 = backend.build_tensor([[0, 1, 0], [2, 0, 3]])
    out = backend.build_tensor([1, 2])
    assert backend.math.all(backend.math.count_nonzero(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cumprod(backend):
    in_1 = backend.build_tensor([1, 2, 3, 4])
    out = backend.build_tensor([1, 2, 6, 24])
    assert backend.math.all(backend.math.cumprod(in_1) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([[1, 2], [3, 8]])
    assert backend.math.all(backend.math.cumprod(in_1, axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cumsum(backend):
    in_1 = backend.build_tensor([1, 2, 3, 4])
    out = backend.build_tensor([1, 3, 6, 10])
    assert backend.math.all(backend.math.cumsum(in_1) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([[1, 2], [4, 6]])
    assert backend.math.all(backend.math.cumsum(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_deg2rad(backend):
    in_1 = backend.build_tensor([0, 90, 180])
    out = backend.build_tensor([0.0, math.pi / 2, math.pi])
    assert backend.math.allclose(backend.math.deg2rad(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_diag(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    out = backend.build_tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert backend.math.all(backend.math.diag(in_1) == out)

    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor([2, 6])
    assert backend.math.all(backend.math.diag(in_1, k=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_diagonal(backend):
    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor([1, 5, 9])
    assert backend.math.all(backend.math.diagonal(in_1) == out)

    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor([2, 6])
    assert backend.math.all(backend.math.diagonal(in_1, offset=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_diff(backend):
    in_1 = backend.build_tensor([1, 2, 4, 7, 0])
    out = backend.build_tensor([1, 2, 3, -7])
    assert backend.math.all(backend.math.diff(in_1) == out)

    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6]])
    out = backend.build_tensor([[1, 1, 1]])
    assert backend.math.all(backend.math.diff(in_1, axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dot(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor(11)  # 1*3 + 2*4
    assert backend.math.all(backend.math.dot(in_1, in_2) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    in_2 = backend.build_tensor([5, 6])
    out = backend.build_tensor([17, 39])  # [1*5+2*6, 3*5+4*6]
    assert backend.math.all(backend.math.dot(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dstack(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor([[[1], [2]], [[3], [4]]])
    assert backend.math.all(backend.math.dstack([in_1, in_2]) == out)

    in_1 = backend.build_tensor([[1], [2]])
    in_2 = backend.build_tensor([[3], [4]])
    out = backend.build_tensor([[[1, 3]], [[2, 4]]])
    assert backend.math.all(backend.math.dstack([in_1, in_2]) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_equal(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor([1, 0, 3])
    out = backend.build_tensor([True, False, True])
    assert backend.math.all(backend.math.equal(in_1, in_2) == out)

    in_1 = backend.build_tensor(5)
    in_2 = backend.build_tensor([5, 6, 5])
    out = backend.build_tensor([True, False, True])
    assert backend.math.all(backend.math.equal(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exp(backend):
    in_1 = backend.build_tensor([0.0, 1.0, 2.0])
    out = backend.build_tensor([math.exp(0.0), math.exp(1.0), math.exp(2.0)])
    assert backend.math.allclose(backend.math.exp(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_expand_dims(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    out = backend.build_tensor([[1, 2, 3]])
    assert backend.math.all(backend.math.expand_dims(in_1, axis=0) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([[[1, 2]], [[3, 4]]])
    assert backend.math.all(backend.math.expand_dims(in_1, axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_flip(backend):
    in_1 = backend.build_tensor([1, 2, 3, 4])
    out = backend.build_tensor([4, 3, 2, 1])
    assert backend.math.all(backend.math.flip(in_1) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([[2, 1], [4, 3]])
    assert backend.math.all(backend.math.flip(in_1, axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_floor(backend):
    in_1 = backend.build_tensor([1.1, 2.9, -3.1, 0.0])
    out = backend.build_tensor([1.0, 2.0, -4.0, 0.0])
    assert backend.math.all(backend.math.floor(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_full(backend):
    shape = (2, 3)
    fill_value = 7
    out = backend.build_tensor([[7, 7, 7], [7, 7, 7]])
    assert backend.math.all(backend.math.full(shape, fill_value) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_full_like(backend):
    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    fill_value = 5
    out = backend.build_tensor([[5, 5], [5, 5]])
    assert backend.math.all(backend.math.full_like(in_1, fill_value) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_greater(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor([0, 2, 4])
    out = backend.build_tensor([True, False, False])
    assert backend.math.all(backend.math.greater(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_greater_equal(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor([0, 2, 4])
    out = backend.build_tensor([True, True, False])
    assert backend.math.all(backend.math.greater_equal(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hstack(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor([1, 2, 3, 4])
    assert backend.math.all(backend.math.hstack([in_1, in_2]) == out)

    in_1 = backend.build_tensor([[1], [2]])
    in_2 = backend.build_tensor([[3], [4]])
    out = backend.build_tensor([[1, 3], [2, 4]])
    assert backend.math.all(backend.math.hstack([in_1, in_2]) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hypot(backend):
    in_1 = backend.build_tensor([3.0, 5.0])
    in_2 = backend.build_tensor([4.0, 12.0])
    out = backend.build_tensor([5.0, 13.0])
    assert backend.math.allclose(backend.math.hypot(in_1, in_2), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_identity(backend):
    out = backend.build_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert backend.math.all(backend.math.identity(3) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isclose(backend):
    in_1 = backend.build_tensor([1.0, 2.0, 3.0])
    in_2 = backend.build_tensor([1.000001, 2.0001, 3.1])
    out = backend.build_tensor([True, False, False])
    assert backend.math.all(backend.math.isclose(in_1, in_2, rtol=1e-5, atol=1e-4) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_allclose(backend):
    in_1 = backend.build_tensor([1.0, 2.0, 3.0])
    in_2 = backend.build_tensor([1.000001, 2.000001, 3.000001])
    assert backend.math.allclose(in_1, in_2, rtol=1e-5, atol=1e-5)

    in_1 = backend.build_tensor([1.0, 2.0, 3.0])
    in_2 = backend.build_tensor([1.000001, 2.0001, 3.000001])
    assert not backend.math.allclose(in_1, in_2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isfinite(backend):
    in_1 = backend.build_tensor([1.0, math.inf, -math.inf, math.nan])
    out = backend.build_tensor([True, False, False, False])
    assert backend.math.all(backend.math.isfinite(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isinf(backend):
    in_1 = backend.build_tensor([1.0, math.inf, -math.inf, math.nan])
    out = backend.build_tensor([False, True, True, False])
    assert backend.math.all(backend.math.isinf(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_isnan(backend):
    in_1 = backend.build_tensor([1.0, math.inf, -math.inf, math.nan])
    out = backend.build_tensor([False, False, False, True])
    assert backend.math.all(backend.math.isnan(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_less(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor([0, 2, 4])
    out = backend.build_tensor([False, False, True])
    assert backend.math.all(backend.math.less(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_less_equal(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor([0, 2, 4])
    out = backend.build_tensor([True, True, False])
    assert backend.math.all(backend.math.greater_equal(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_log(backend):
    in_1 = backend.build_tensor([1.0, math.e, math.e**2])
    out = backend.build_tensor([0.0, 1.0, 2.0])
    assert backend.math.allclose(backend.math.log(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logical_and(backend):
    in_1 = backend.build_tensor([True, True, False, False])
    in_2 = backend.build_tensor([True, False, True, False])
    out = backend.build_tensor([True, False, False, False])
    assert backend.math.all(backend.math.logical_and(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logical_not(backend):
    in_1 = backend.build_tensor([True, False, True])
    out = backend.build_tensor([False, True, False])
    assert backend.math.all(backend.math.logical_not(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logical_or(backend):
    in_1 = backend.build_tensor([True, True, False, False])
    in_2 = backend.build_tensor([True, False, True, False])
    out = backend.build_tensor([True, True, True, False])
    assert backend.math.all(backend.math.logical_or(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logical_xor(backend):
    in_1 = backend.build_tensor([True, True, False, False])
    in_2 = backend.build_tensor([True, False, True, False])
    out = backend.build_tensor([False, True, True, False])
    assert backend.math.all(backend.math.logical_xor(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_maximum(backend):
    in_1 = backend.build_tensor([1, 5, 2])
    in_2 = backend.build_tensor([3, 2, 8])
    out = backend.build_tensor([3, 5, 8])
    assert backend.math.all(backend.math.maximum(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_median(backend):
    in_1 = backend.build_tensor([1, 5, 2, 8, 3])
    out = backend.build_tensor(3.0)
    assert backend.math.all(backend.math.median(in_1) == out)

    in_1 = backend.build_tensor([[1, 5], [3, 2]])
    out = backend.build_tensor([2.0, 3.5])
    assert backend.math.all(backend.math.median(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_meshgrid(backend):
    x = backend.build_tensor([1, 2])
    y = backend.build_tensor([3, 4])
    X, Y = backend.math.meshgrid(x, y)
    out_X = backend.build_tensor([[1, 1], [2, 2]])
    out_Y = backend.build_tensor([[3, 4], [3, 4]])
    assert backend.math.all(X == out_X)
    assert backend.math.all(Y == out_Y)


@pytest.mark.parametrize("backend", BACKENDS)
def test_minimum(backend):
    in_1 = backend.build_tensor([1, 5, 2])
    in_2 = backend.build_tensor([3, 2, 8])
    out = backend.build_tensor([1, 2, 2])
    assert backend.math.all(backend.math.minimum(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mod(backend):
    in_1 = backend.build_tensor([7, 8, 9])
    in_2 = backend.build_tensor([3, 3, 2])
    out = backend.build_tensor([1, 2, 1])
    assert backend.math.all(backend.math.mod(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_moveaxis(backend):
    in_1 = backend.build_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # shape (2,2,2)
    out = backend.build_tensor([[[1, 5], [3, 7]], [[2, 6], [4, 8]]])  # shape (2,2,2)
    assert backend.math.all(backend.math.moveaxis(in_1, 0, -1) == out)

    in_1 = backend.build_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    out = backend.build_tensor([[[1, 3], [2, 4]], [[5, 7], [6, 8]]])
    assert backend.math.all(backend.math.moveaxis(in_1, 1, 0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nanargmax(backend):
    in_1 = backend.build_tensor([1, 5, math.nan, 8, 3])
    out = backend.build_tensor(3)
    assert backend.math.all(backend.math.nanargmax(in_1) == out)

    in_1 = backend.build_tensor([[1, math.nan, 2], [8, 3, 4]])
    out = backend.build_tensor([1, 0])
    assert backend.math.all(backend.math.nanargmax(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nanargmin(backend):
    in_1 = backend.build_tensor([math.nan, 5, 2, 8, 3])
    out = backend.build_tensor(2)
    assert backend.math.all(backend.math.nanargmin(in_1) == out)

    in_1 = backend.build_tensor([[math.nan, 5, 2], [8, 3, 4]])
    out = backend.build_tensor([1, 1])
    assert backend.math.all(backend.math.nanargmin(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nanmax(backend):
    in_1 = backend.build_tensor([1, 5, math.nan, 8, 3])
    out = backend.build_tensor(8)
    assert backend.math.all(backend.math.nanmax(in_1) == out)

    in_1 = backend.build_tensor([[1, math.nan, 2], [8, 3, 4]])
    out = backend.build_tensor([8, 3, 4])
    assert backend.math.all(backend.math.nanmax(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nanmean(backend):
    in_1 = backend.build_tensor([1.0, 2.0, math.nan, 4.0])
    out = backend.build_tensor(7.0 / 3.0)
    assert backend.math.allclose(backend.math.nanmean(in_1), out)

    in_1 = backend.build_tensor([[1.0, math.nan], [3.0, 4.0]])
    out = backend.build_tensor([2.0, 4.0])
    assert backend.math.allclose(backend.math.nanmean(in_1, axis=0), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nanmin(backend):
    in_1 = backend.build_tensor([1, 5, math.nan, 8, 3])
    out = backend.build_tensor(1)
    assert backend.math.all(backend.math.nanmin(in_1) == out)

    in_1 = backend.build_tensor([[1, math.nan, 2], [8, 3, 4]])
    out = backend.build_tensor([1, 3, 2])
    assert backend.math.all(backend.math.nanmin(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nansum(backend):
    in_1 = backend.build_tensor([1.0, 2.0, math.nan, 4.0])
    out = backend.build_tensor(7.0)
    assert backend.math.allclose(backend.math.nansum(in_1), out)

    in_1 = backend.build_tensor([[1.0, math.nan], [3.0, 4.0]])
    out = backend.build_tensor([4.0, 4.0])
    assert backend.math.allclose(backend.math.nansum(in_1, axis=0), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nan_to_num(backend):
    in_1 = backend.build_tensor([1.0, math.nan, math.inf, -math.inf])
    # Explicitly set posinf and neginf to match expected output for consistency
    out = backend.build_tensor([1.0, 0.0, 1e18, -1e18])
    assert backend.math.allclose(
        backend.math.nan_to_num(in_1, nan=0.0, posinf=1e18, neginf=-1e18), out
    )

    in_1 = backend.build_tensor([1.0, math.nan, math.inf])
    out = backend.build_tensor([1.0, 5.0, 10.0])
    assert backend.math.allclose(backend.math.nan_to_num(in_1, nan=5.0, posinf=10.0), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ndim(backend):
    in_1 = backend.build_tensor(1)
    assert backend.math.ndim(in_1) == 0

    in_1 = backend.build_tensor([1, 2, 3])
    assert backend.math.ndim(in_1) == 1

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    assert backend.math.ndim(in_1) == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_nonzero(backend):
    in_1 = backend.build_tensor([0, 1, 0, 2, 0, 3])
    out = backend.build_tensor([1, 3, 5])
    # Nonzero returns a tuple of arrays, one for each dimension.
    # For 1D, it's (array([1, 3, 5]),)
    assert backend.math.all(backend.math.nonzero(in_1)[0] == out)

    in_1 = backend.build_tensor([[0, 1, 0], [2, 0, 3]])
    # Expected: (array([0, 1, 1]), array([1, 0, 2]))
    rows, cols = backend.math.nonzero(in_1)
    assert backend.math.all(rows == backend.build_tensor([0, 1, 1]))
    assert backend.math.all(cols == backend.build_tensor([1, 0, 2]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_not_equal(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    in_2 = backend.build_tensor([1, 0, 3])
    out = backend.build_tensor([False, True, False])
    assert backend.math.all(backend.math.not_equal(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_outer(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4, 5])
    out = backend.build_tensor([[3, 4, 5], [6, 8, 10]])
    assert backend.math.all(backend.math.outer(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pad(backend):
    in_1 = backend.build_tensor([1, 2, 3])
    pad_width = ((1, 2),)  # 1 before, 2 after
    out = backend.build_tensor([0, 1, 2, 3, 0, 0])
    assert backend.math.all(backend.math.pad(in_1, pad_width) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    pad_width = ((1, 1), (0, 1))  # 1 row top/bottom, 0 col left, 1 col right
    out = backend.build_tensor([[0, 0, 0], [1, 2, 0], [3, 4, 0], [0, 0, 0]])
    assert backend.math.all(backend.math.pad(in_1, pad_width) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_prod(backend):
    in_1 = backend.build_tensor([1, 2, 3, 4])
    out = backend.build_tensor(24)
    assert backend.math.all(backend.math.prod(in_1) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([3, 8])
    assert backend.math.all(backend.math.prod(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ptp(backend):
    in_1 = backend.build_tensor([1, 5, 2, 8, 3])
    out = backend.build_tensor(7)  # 8 - 1
    assert backend.math.all(backend.math.ptp(in_1) == out)

    in_1 = backend.build_tensor([[1, 5], [3, 2]])
    out = backend.build_tensor([2, 3])  # [3-1, 5-2]
    assert backend.math.all(backend.math.ptp(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ravel(backend):
    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([1, 2, 3, 4])
    assert backend.math.all(backend.math.ravel(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_real(backend):
    # Assuming complex numbers are supported by the backend
    in_1 = backend.build_tensor([1 + 2j, 3 - 4j, 5])
    out = backend.build_tensor([1, 3, 5])
    assert backend.math.all(backend.math.real(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reciprocal(backend):
    in_1 = backend.build_tensor([1.0, 2.0, 0.5])
    out = backend.build_tensor([1.0, 0.5, 2.0])
    assert backend.math.allclose(backend.math.reciprocal(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_repeat(backend):
    in_1 = backend.build_tensor([1, 2])
    out = backend.build_tensor([1, 1, 1, 2, 2, 2])
    assert backend.math.all(backend.math.repeat(in_1, 3) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([[1, 1, 2, 2], [3, 3, 4, 4]])
    assert backend.math.all(backend.math.repeat(in_1, 2, axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reshape(backend):
    in_1 = backend.build_tensor([1, 2, 3, 4, 5, 6])
    out = backend.build_tensor([[1, 2, 3], [4, 5, 6]])
    assert backend.math.all(backend.math.reshape(in_1, (2, 3)) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4], [5, 6]])
    out = backend.build_tensor([1, 2, 3, 4, 5, 6])
    assert backend.math.all(backend.math.reshape(in_1, -1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_roll(backend):
    in_1 = backend.build_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    out = backend.build_tensor([7, 8, 9, 0, 1, 2, 3, 4, 5, 6])
    assert backend.math.all(backend.math.roll(in_1, 3) == out)

    in_1 = backend.build_tensor([[0, 1], [2, 3]])
    out = backend.build_tensor([[1, 0], [3, 2]])
    assert backend.math.all(backend.math.roll(in_1, 1, axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sign(backend):
    in_1 = backend.build_tensor([-2, 0, 5])
    out = backend.build_tensor([-1, 0, 1])
    assert backend.math.all(backend.math.sign(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sin(backend):
    in_1 = backend.build_tensor([0.0, math.pi / 2, math.pi])
    out = backend.build_tensor([0.0, 1.0, 0.0])
    assert backend.math.allclose(backend.math.sin(in_1), out, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sinh(backend):
    in_1 = backend.build_tensor([0.0, 1.0])
    out = backend.build_tensor([math.sinh(0.0), math.sinh(1.0)])
    assert backend.math.allclose(backend.math.sinh(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_size(backend):
    in_1 = backend.build_tensor(5)
    assert backend.math.size(in_1) == 1

    in_1 = backend.build_tensor([1, 2, 3])
    assert backend.math.size(in_1) == 3

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    assert backend.math.size(in_1) == 4


@pytest.mark.parametrize("backend", BACKENDS)
def test_sort(backend):
    in_1 = backend.build_tensor([3, 1, 4, 1, 5, 9, 2, 6])
    out = backend.build_tensor([1, 1, 2, 3, 4, 5, 6, 9])
    assert backend.math.all(backend.math.sort(in_1) == out)

    in_1 = backend.build_tensor([[0, 3], [2, 1]])
    out = backend.build_tensor([[0, 3], [1, 2]])
    assert backend.math.all(backend.math.sort(in_1, axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_split(backend):
    in_1 = backend.build_tensor([1, 2, 3, 4, 5, 6])
    out_1 = backend.build_tensor([1, 2])
    out_2 = backend.build_tensor([3, 4])
    res = backend.math.split(in_1, 2)
    assert len(res) == 3
    assert backend.math.all(res[0] == out_1)
    assert backend.math.all(res[1] == out_2)

    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6]])
    out_1 = backend.build_tensor([[1], [4]])
    out_2 = backend.build_tensor([[2], [5]])
    out_3 = backend.build_tensor([[3], [6]])
    res = backend.math.split(in_1, 1, axis=1)
    assert len(res) == 3
    assert backend.math.all(res[0] == out_1)
    assert backend.math.all(res[1] == out_2)
    assert backend.math.all(res[2] == out_3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_stack(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor([[1, 2], [3, 4]])
    assert backend.math.all(backend.math.stack([in_1, in_2]) == out)

    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor([[1, 3], [2, 4]])
    assert backend.math.all(backend.math.stack([in_1, in_2], axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_std(backend):
    in_1 = backend.build_tensor([1.0, 2.0, 3.0])
    out = backend.build_tensor(1.0)
    assert backend.math.allclose(backend.math.std(in_1), out)

    in_1 = backend.build_tensor([[1.0, 2.0], [3.0, 4.0]])
    out = backend.build_tensor([math.sqrt(2.0), math.sqrt(2.0)])
    assert backend.math.allclose(backend.math.std(in_1, axis=0), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sum(backend):
    in_1 = backend.build_tensor([1, 2, 3, 4])
    out = backend.build_tensor(10)
    assert backend.math.all(backend.math.sum(in_1) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([4, 6])
    assert backend.math.all(backend.math.sum(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_swapaxes(backend):
    in_1 = backend.build_tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])  # (2,2,2)
    out = backend.build_tensor([[[0, 4], [2, 6]], [[1, 5], [3, 7]]])  # (2,2,2)
    assert backend.math.all(backend.math.swapaxes(in_1, 0, 2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_take(backend):
    in_1 = backend.build_tensor([4, 3, 5, 7, 6, 8])
    indices = backend.build_tensor([0, 1, 4])
    out = backend.build_tensor([4, 3, 6])
    assert backend.math.all(backend.math.take(in_1, indices) == out)

    in_1 = backend.build_tensor([[0, 1], [2, 3]])
    indices = backend.build_tensor([1, 0])
    out = backend.build_tensor([[2, 3], [0, 1]])
    assert backend.math.all(backend.math.take(in_1, indices, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tan(backend):
    in_1 = backend.build_tensor([0.0, math.pi / 4])
    out = backend.build_tensor([0.0, 1.0])
    assert backend.math.allclose(backend.math.tan(in_1), out, atol=1e-7)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tanh(backend):
    in_1 = backend.build_tensor([0.0, 1.0])
    out = backend.build_tensor([math.tanh(0.0), math.tanh(1.0)])
    assert backend.math.allclose(backend.math.tanh(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensordot(backend):
    in_1 = backend.build_tensor([[1, 2], [3, 4]])  # (2,2)
    in_2 = backend.build_tensor([[10, 20], [30, 40]])  # (2,2)
    out = backend.build_tensor(300)  # sum(A*B)
    assert backend.math.all(backend.math.tensordot(in_1, in_2, axes=2) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    in_2 = backend.build_tensor([[10, 20], [30, 40]])
    out = backend.build_tensor([[70, 100], [150, 220]])  # A @ B
    assert backend.math.all(backend.math.tensordot(in_1, in_2, axes=([1], [0])) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_round(backend):
    in_1 = backend.build_tensor([1.23, 1.78, 2.5, 3.14159])
    out = backend.build_tensor([1.0, 2.0, 2.0, 3.0])
    assert backend.math.all(backend.math.round(in_1) == out)

    in_1 = backend.build_tensor([3.14159, 2.71828])
    out = backend.build_tensor([3.14, 2.72])
    assert backend.math.all(backend.math.round(in_1, decimals=2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tile(backend):
    in_1 = backend.build_tensor([0, 1, 2])
    out = backend.build_tensor([0, 1, 2, 0, 1, 2])
    assert backend.math.all(backend.math.tile(in_1, 2) == out)

    in_1 = backend.build_tensor([[0, 1], [2, 3]])
    out = backend.build_tensor([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]])
    assert backend.math.all(backend.math.tile(in_1, (2, 2)) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_trace(backend):
    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor(15)  # 1+5+9
    assert backend.math.all(backend.math.trace(in_1) == out)

    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor(8)  # 2+6
    assert backend.math.all(backend.math.trace(in_1, offset=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tril(backend):
    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
    assert backend.math.all(backend.math.tril(in_1) == out)

    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor([[1, 2, 0], [4, 5, 6], [7, 8, 9]])
    assert backend.math.all(backend.math.tril(in_1, k=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_triu(backend):
    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor([[1, 2, 3], [0, 5, 6], [0, 0, 9]])
    assert backend.math.all(backend.math.triu(in_1) == out)

    in_1 = backend.build_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = backend.build_tensor([[0, 0, 3], [0, 0, 0], [0, 0, 0]])
    assert backend.math.all(backend.math.triu(in_1, k=2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_trunc(backend):
    in_1 = backend.build_tensor([1.23, 1.78, -2.5, 3.14159])
    out = backend.build_tensor([1.0, 1.0, -2.0, 3.0])
    assert backend.math.all(backend.math.trunc(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vdot(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor(11)  # 1*3 + 2*4
    assert backend.math.all(backend.math.vdot(in_1, in_2) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    in_2 = backend.build_tensor([[5, 6], [7, 8]])
    out = backend.build_tensor(70)  # 1*5 + 2*6 + 3*7 + 4*8
    assert backend.math.all(backend.math.vdot(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inner(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor(11)  # 1*3 + 2*4
    assert backend.math.all(backend.math.inner(in_1, in_2) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    in_2 = backend.build_tensor([5, 6])
    out = backend.build_tensor([17, 39])  # [1*5+2*6, 3*5+4*6]
    assert backend.math.all(backend.math.inner(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vstack(backend):
    in_1 = backend.build_tensor([1, 2])
    in_2 = backend.build_tensor([3, 4])
    out = backend.build_tensor([[1, 2], [3, 4]])
    assert backend.math.all(backend.math.vstack([in_1, in_2]) == out)

    in_1 = backend.build_tensor([[1], [2]])
    in_2 = backend.build_tensor([[3], [4]])
    out = backend.build_tensor([[1], [2], [3], [4]])
    assert backend.math.all(backend.math.vstack([in_1, in_2]) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_where(backend):
    condition = backend.build_tensor([True, False, True])
    x = backend.build_tensor([1, 2, 3])
    y = backend.build_tensor([10, 20, 30])
    out = backend.build_tensor([1, 20, 3])
    assert backend.math.all(backend.math.where(condition, x, y) == out)

    condition = backend.build_tensor([True, False, True])
    x = backend.build_tensor([1, 2, 3])
    out = backend.build_tensor([1, 0, 3])  # default y is 0
    assert backend.math.all(backend.math.where(condition, x) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_divide(backend):
    in_1 = backend.build_tensor([10.0, 4.0, 9.0])
    in_2 = backend.build_tensor([2.0, 2.0, 3.0])
    out = backend.build_tensor([5.0, 2.0, 3.0])
    assert backend.math.allclose(backend.math.divide(in_1, in_2), out)

    in_1 = backend.build_tensor([10.0, 4.0, 9.0])
    in_2 = backend.build_tensor(2.0)
    out = backend.build_tensor([5.0, 2.0, 4.5])
    assert backend.math.allclose(backend.math.divide(in_1, in_2), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_power(backend):
    in_1 = backend.build_tensor([2, 3, 4])
    in_2 = backend.build_tensor([2, 3, 1])
    out = backend.build_tensor([4, 27, 4])
    assert backend.math.all(backend.math.power(in_1, in_2) == out)

    in_1 = backend.build_tensor([2, 3, 4])
    in_2 = backend.build_tensor(2)
    out = backend.build_tensor([4, 9, 16])
    assert backend.math.all(backend.math.power(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_negative(backend):
    in_1 = backend.build_tensor([1, -2, 0])
    out = backend.build_tensor([-1, 2, 0])
    assert backend.math.all(backend.math.negative(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_square(backend):
    in_1 = backend.build_tensor([1, -2, 3])
    out = backend.build_tensor([1, 4, 9])
    assert backend.math.all(backend.math.square(in_1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sqrt(backend):
    in_1 = backend.build_tensor([1.0, 4.0, 9.0])
    out = backend.build_tensor([1.0, 2.0, 3.0])
    assert backend.math.allclose(backend.math.sqrt(in_1), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_squeeze(backend):
    in_1 = backend.build_tensor([[[1, 2]]])  # shape (1,1,2)
    out = backend.build_tensor([1, 2])  # shape (2,)
    assert backend.math.all(backend.math.squeeze(in_1) == out)

    in_1 = backend.build_tensor([[[1, 2], [3, 4]]])  # shape (1,2,2)
    out = backend.build_tensor([[1, 2], [3, 4]])  # shape (2,2)
    assert backend.math.all(backend.math.squeeze(in_1, axis=0) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unsqueeze(backend):
    in_1 = backend.build_tensor([1, 2, 3])  # shape (3,)
    out = backend.build_tensor([[1, 2, 3]])  # shape (1,3)
    assert backend.math.all(backend.math.unsqueeze(in_1, axis=0) == out)

    in_1 = backend.build_tensor([[1, 2], [3, 4]])  # shape (2,2)
    out = backend.build_tensor([[[1, 2]], [[3, 4]]])  # shape (2,1,2)
    assert backend.math.all(backend.math.unsqueeze(in_1, axis=1) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_transpose(backend):
    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([[1, 3], [2, 4]])
    assert backend.math.all(backend.math.transpose(in_1) == out)

    in_1 = backend.build_tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    out = backend.build_tensor([[[0, 2], [1, 3]], [[4, 6], [5, 7]]])
    assert backend.math.all(backend.math.transpose(in_1, axes=(0, 2, 1)) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_var(backend):
    in_1 = backend.build_tensor([1.0, 2.0, 3.0])
    out = backend.build_tensor(2.0 / 3.0)
    assert backend.math.allclose(backend.math.var(in_1), out)

    in_1 = backend.build_tensor([[1.0, 2.0], [3.0, 4.0]])
    out = backend.build_tensor([1.0, 1.0])
    assert backend.math.allclose(backend.math.var(in_1, axis=0), out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_floor_divide(backend):
    in_1 = backend.build_tensor([7, 8, -7])
    in_2 = backend.build_tensor([3, 3, 3])
    out = backend.build_tensor([2, 2, -3])
    assert backend.math.all(backend.math.floor_divide(in_1, in_2) == out)

    in_1 = backend.build_tensor([7, 8, -7])
    in_2 = backend.build_tensor(3)
    out = backend.build_tensor([2, 2, -3])
    assert backend.math.all(backend.math.floor_divide(in_1, in_2) == out)


@pytest.mark.parametrize("backend", BACKENDS)
def test_flatten(backend):
    in_1 = backend.build_tensor([[1, 2], [3, 4]])
    out = backend.build_tensor([1, 2, 3, 4])
    assert backend.math.all(backend.math.flatten(in_1) == out)

    in_1 = backend.build_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2,2,2)
    out = backend.build_tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # (2,4)
    assert backend.math.all(backend.math.flatten(in_1, start_dim=1) == out)
