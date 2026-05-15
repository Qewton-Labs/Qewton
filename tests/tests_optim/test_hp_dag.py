import random
import pytest

from pioneer.optim.parameters.dag import HyperParameterDAG
from pioneer.optim.parameters.hyperparameter_base import HyperParameter


class DummyHP(HyperParameter):
    """Minimal implementation of HyperParameter for DAG testing."""

    def __init__(self, name, dependencies=None, active=True, grid=None, random_val=0):
        self._dependencies = dependencies or set()
        self._active = active
        self._grid = grid or [0, 1]
        self._random_val = random_val
        super().__init__(parameter_range=(0, 1), initial_value=0, name=name)

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def tuning_grid(self):
        return self._grid

    def is_active(self, config=None):
        if callable(self._active):
            return self._active(config)
        return self._active

    def sample_parameter_random(self):
        return self._random_val

    def sample_from_unit(self, x: float):
        return self.sample_parameter_random()


# --- Initialization & Sorting Tests ---


def test_dag_init_empty():
    dag = HyperParameterDAG(set())
    assert not dag.sorted_nodes
    assert not dag.graph


def test_dag_init_single():
    h = DummyHP("a")
    dag = HyperParameterDAG({h})
    assert dag.sorted_nodes == [h]


def test_dag_duplicate_names():
    h1 = DummyHP("x")
    h2 = DummyHP("x")
    with pytest.raises(
        ValueError, match="Found at least two HyperParameters with the name 'x'"
    ):
        HyperParameterDAG({h1, h2})


def test_dag_build_graph_structure():
    h1 = DummyHP("parent")
    h2 = DummyHP("child", dependencies={h1})
    dag = HyperParameterDAG({h1, h2})
    assert h2 in dag.graph[h1]
    assert dag.graph[h2] == set()


def test_dag_sort_linear():
    h1 = DummyHP("a")
    h2 = DummyHP("b", dependencies={h1})
    h3 = DummyHP("c", dependencies={h2})
    dag = HyperParameterDAG({h1, h2, h3})
    assert dag.sorted_nodes == [h1, h2, h3]


def test_dag_sort_parallel():
    h1 = DummyHP("root")
    h2 = DummyHP("a", dependencies={h1})
    h3 = DummyHP("b", dependencies={h1})
    dag = HyperParameterDAG({h1, h2, h3})
    assert h1 is dag.sorted_nodes[0]
    assert set(dag.sorted_nodes[1:]) == {h2, h3}


def test_dag_cycle_self():
    h1 = DummyHP("a")
    h1._dependencies = {h1}
    with pytest.raises(ValueError, match="Cycle detected"):
        HyperParameterDAG({h1})


def test_dag_cycle_mutual():
    h1 = DummyHP("a")
    h2 = DummyHP("b", dependencies={h1})
    h1._dependencies = {h2}
    with pytest.raises(ValueError, match="Cycle detected"):
        HyperParameterDAG({h1, h2})


def test_dag_cycle_deep():
    h1 = DummyHP("a")
    h2 = DummyHP("b", dependencies={h1})
    h3 = DummyHP("c", dependencies={h2})
    h1._dependencies = {h3}
    with pytest.raises(ValueError, match="Cycle detected"):
        HyperParameterDAG({h1, h2, h3})


def test_dag_disconnected_components():
    h1 = DummyHP("a1")
    h2 = DummyHP("a2", dependencies={h1})
    h3 = DummyHP("b1")
    h4 = DummyHP("b2", dependencies={h3})
    dag = HyperParameterDAG({h1, h2, h3, h4})
    # Check that relative ordering within components is preserved
    order = dag.sorted_nodes
    assert len(order) == 4
    idx_1 = [i for i in range(4) if order[i].name == h1.name][0]
    idx_2 = [i for i in range(4) if order[i].name == h2.name][0]
    assert idx_1 < idx_2
    idx_3 = [i for i in range(4) if order[i].name == h3.name][0]
    idx_4 = [i for i in range(4) if order[i].name == h4.name][0]
    assert idx_3 < idx_4


# --- Random Sampling Tests ---


def test_random_samples_count():
    h1 = DummyHP("a")
    dag = HyperParameterDAG({h1})
    samples = dag.create_random_samples(7)
    assert len(samples) == 7


def test_random_samples_keys():
    h1 = DummyHP("a")
    h2 = DummyHP("b")
    dag = HyperParameterDAG({h1, h2})
    sample = dag.create_random_samples(1)[0]
    assert set(sample.keys()) == {"a", "b"}


def test_random_samples_inactive_skipped():
    h1 = DummyHP("a", random_val=0)
    # b is inactive if a is 0
    h2 = DummyHP("b", dependencies={h1}, active=lambda c: c["a"] != 0)
    dag = HyperParameterDAG({h1, h2})
    samples = dag.create_random_samples(1)
    assert "a" in samples[0]
    assert "b" not in samples[0]


def test_random_samples_active_included():
    h1 = DummyHP("a", random_val=1)
    h2 = DummyHP("b", dependencies={h1}, active=lambda c: c["a"] == 1, random_val=99)
    dag = HyperParameterDAG({h1, h2})
    samples = dag.create_random_samples(1)
    assert samples[0]["a"] == 1
    assert samples[0]["b"] == 99


def test_random_samples_all_inactive():
    h1 = DummyHP("a", active=False)
    dag = HyperParameterDAG({h1})
    samples = dag.create_random_samples(1)
    assert samples[0] == {}


def test_random_samples_complex_dependency_chain():
    h1 = DummyHP("h1", random_val=1)
    h2 = DummyHP("h2", dependencies={h1}, active=lambda c: c["h1"] == 1, random_val=2)
    h3 = DummyHP("h3", dependencies={h2}, active=lambda c: c.get("h2") == 2, random_val=3)
    dag = HyperParameterDAG({h1, h2, h3})
    sample = dag.create_random_samples(1)[0]
    assert sample == {"h1": 1, "h2": 2, "h3": 3}


# --- Grid Sampling Tests ---


def test_grid_samples_basic():
    h1 = DummyHP("a", grid=[1, 2])
    h2 = DummyHP("b", grid=[10])
    dag = HyperParameterDAG({h1, h2})
    with pytest.warns(UserWarning):
        samples = dag.create_grid_samples(10)
    assert len(samples) == 2
    assert {"a": 1, "b": 10} in samples
    assert {"a": 2, "b": 10} in samples


def test_grid_samples_single_value():
    h1 = DummyHP("a", grid=[5])
    dag = HyperParameterDAG({h1})
    with pytest.warns(UserWarning):
        grid = dag.create_grid_samples(10)
    assert grid == [{"a": 5}]


def test_grid_samples_limit_applied():
    h1 = DummyHP("a", grid=list(range(100)))
    dag = HyperParameterDAG({h1})
    samples = dag.create_grid_samples(20)
    assert len(samples) == 20


def test_grid_samples_resampling_deterministic():
    h1 = DummyHP("a", grid=list(range(1000)))
    dag = HyperParameterDAG({h1})
    random.seed(123)
    s1 = dag.create_grid_samples(10)
    random.seed(123)
    s2 = dag.create_grid_samples(10)
    assert s1 == s2


def test_grid_samples_conditional_active():
    h1 = DummyHP("a", grid=[0, 1])
    # b active only if a=1
    h2 = DummyHP("b", dependencies={h1}, grid=[100], active=lambda c: c.get("a") == 1)
    dag = HyperParameterDAG({h1, h2})
    with pytest.warns(UserWarning):
        samples = dag.create_grid_samples(10)
    # Combos from product: (a=0, b=100), (a=1, b=100)
    # Resulting configs after activity check: {"a": 0}, {"a": 1, "b": 100}
    assert len(samples) == 2
    assert {"a": 0} in samples
    assert {"a": 1, "b": 100} in samples


def test_grid_samples_conditional_inactive():
    h1 = DummyHP("a", grid=[1])
    h2 = DummyHP("b", dependencies={h1}, active=False, grid=[10, 20])
    dag = HyperParameterDAG({h1, h2})
    with pytest.warns(UserWarning):
        samples = dag.create_grid_samples(10)
    # Grid has 2 combos, but b is always inactive
    assert samples == [{"a": 1}, {"a": 1}]


def test_grid_samples_multiple_layers():
    h1 = DummyHP("a", grid=[1])
    h2 = DummyHP("b", dependencies={h1}, grid=[2])
    h3 = DummyHP("c", dependencies={h2}, grid=[3])
    dag = HyperParameterDAG({h1, h2, h3})
    with pytest.warns(UserWarning):
        grid = dag.create_grid_samples(10)
    assert grid == [{"a": 1, "b": 2, "c": 3}]


# --- Edge Case & Regression Tests ---


def test_dag_shared_parent():
    h1 = DummyHP("root")
    h2 = DummyHP("a", dependencies={h1})
    h3 = DummyHP("b", dependencies={h1})
    h4 = DummyHP("leaf", dependencies={h2, h3})
    dag = HyperParameterDAG({h1, h2, h3, h4})
    order = dag.sorted_nodes
    assert order[0].name == h1.name
    assert order[-1].name == h4.name


def test_dag_complex_activity_chain():
    # h1 is always 1
    # h2 active if h1=1 -> h2 is 2
    # h3 active if h2=2 -> h3 is 3
    h1 = DummyHP("h1", grid=[1])
    h2 = DummyHP("h2", dependencies={h1}, grid=[2], active=lambda c: c.get("h1") == 1)
    h3 = DummyHP("h3", dependencies={h2}, grid=[3], active=lambda c: c.get("h2") == 2)
    dag = HyperParameterDAG({h1, h2, h3})
    with pytest.warns(UserWarning):
        grid = dag.create_grid_samples(10)

    assert grid == [{"h1": 1, "h2": 2, "h3": 3}]


def test_dag_grid_with_size_zero_range():
    h1 = DummyHP("a", grid=[])
    dag = HyperParameterDAG({h1})
    # product(*) with an empty list results in empty list
    with pytest.warns(UserWarning):
        samples = dag.create_grid_samples(10)
    assert len(samples) == 2


def test_dag_random_samples_zero_request():
    h1 = DummyHP("a")
    dag = HyperParameterDAG({h1})
    assert not dag.create_random_samples(0)


def test_dag_grid_samples_zero_request():
    h1 = DummyHP("a", grid=[1, 2])
    dag = HyperParameterDAG({h1})
    assert not dag.create_grid_samples(0)
