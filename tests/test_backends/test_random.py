import inspect
import math
import pytest

from qewton.backends.base import ComputingBackend


def all_subclasses(cls):
    result = []
    for sub_cls in cls.__subclasses__():
        if not inspect.isabstract(sub_cls) and hasattr(sub_cls, "random"):
            result.append(sub_cls)
        result.extend(all_subclasses(sub_cls))
    return result


BACKENDS = all_subclasses(ComputingBackend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_set_seed(backend):
    backend.random.set_seed(42)


@pytest.mark.parametrize("backend", BACKENDS)
def test_uniform(backend):
    uniform_points = backend.random.uniform((20, 20, 1))
    assert uniform_points.shape == (20, 20, 1)
    uniform_points = backend.random.uniform(10, -10, 10)
    for p in uniform_points:
        assert p >= -10 and p <= 10


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal_and_standard_normal(backend):
    # Normal with mean/std: check shape and finite values
    a = backend.random.normal((100,))
    assert a.shape == (100,)
    for v in a:
        assert math.isfinite(float(v))

    b = backend.random.normal((50,), mean=5.0, std=2.0)
    assert b.shape == (50,)
    for v in b:
        assert math.isfinite(float(v))

    # standard_normal
    s = backend.random.standard_normal((80,))
    assert s.shape == (80,)
    for v in s:
        assert math.isfinite(float(v))


@pytest.mark.parametrize("backend", BACKENDS)
def test_randint_and_choice(backend):
    # randint: single bound
    r = backend.random.randint(0, 10, shape=(100,))
    assert r.shape == (100,)
    for v in r:
        iv = int(v)
        assert 0 <= iv < 10

    # choice from integer
    c = backend.random.choice(5, shape=(50,))
    assert c.shape == (50,)
    for v in c:
        iv = int(v)
        assert 0 <= iv < 5

    # choice from list
    pool = [10, 20, 30]
    c2 = backend.random.choice(pool, shape=(30,))
    assert c2.shape == (30,)
    for v in c2:
        assert int(v) in pool


@pytest.mark.parametrize("backend", BACKENDS)
def test_permutation_and_shuffle(backend):
    # permutation of integer range
    p = backend.random.permutation(5)
    assert len(p) == 5
    assert sorted([int(x) for x in p]) == list(range(5))

    # shuffle should preserve elements
    seq = list(range(10))
    orig = seq.copy()
    # try shuffle on a python list; implementations may also accept tensors
    seq = backend.build_tensor(seq)
    backend.random.shuffle(seq)
    assert sorted(seq) == sorted(orig)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exponential_and_multivariate_normal(backend):
    e = backend.random.exponential((60,))
    assert e.shape == (60,)
    for v in e:
        assert float(v) >= 0 and math.isfinite(float(v))

    # multivariate normal: mean length 2, ask for 3 samples
    mean = [0.0, 0.0]
    cov = [[1.0, 0.0], [0.0, 1.0]]
    mv = backend.random.multivariate_normal(mean, cov, shape=3)
    # expect shape (3, 2) or (3,2)-like; check first dim and second equals len(mean)
    assert mv.shape[0] == 3
    assert mv.shape[1] == len(mean)


@pytest.mark.parametrize("backend", BACKENDS)
def test_discrete_and_count_distributions(backend):
    # binomial
    b = backend.random.binomial(10, 0.5, shape=(80,))
    assert b.shape == (80,)
    for v in b:
        iv = int(v)
        assert 0 <= iv <= 10

    # poisson
    p = backend.random.poisson(3.0, shape=(80,))
    assert p.shape == (80,)
    for v in p:
        assert int(v) >= 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_gamma_beta_lognormal_gumbel(backend):
    # gamma
    g = backend.random.gamma(2.0, 1.0, shape=(60,))
    assert g.shape == (60,)
    for v in g:
        assert float(v) >= 0 and math.isfinite(float(v))

    # beta in [0,1]
    be = backend.random.beta(2.0, 5.0, shape=(60,))
    assert be.shape == (60,)
    for v in be:
        fv = float(v)
        assert 0.0 <= fv <= 1.0 and math.isfinite(fv)

    # lognormal > 0
    ln = backend.random.lognormal(mean=0.0, sigma=1.0, shape=(60,))
    assert ln.shape == (60,)
    for v in ln:
        assert float(v) > 0 and math.isfinite(float(v))

    # gumbel finite
    gu = backend.random.gumbel(loc=0.0, scale=1.0, shape=(60,))
    assert gu.shape == (60,)
    for v in gu:
        assert math.isfinite(float(v))
