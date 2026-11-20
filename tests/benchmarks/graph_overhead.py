"""Time of building and running graphs, where bookkeeping dominates.

Uses tiny tensors on the CPU, so the time is spent in Qewton rather than
in the numerics: a PINN pipeline, and an FCN inside 0, 2 and 4 wrapping
GraphNodes. Run with ``python tests/benchmarks/graph_overhead.py``; the
results are logged. Compare a branch with `main` on the same machine.
"""

import logging
import os
import time
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.filterwarnings("ignore")

from qewton.algorithms import FCN  # noqa: E402
from qewton.config import Variable  # noqa: E402
from qewton.data.dataloaders.sampler.random_sampler import RandomUniformSampler  # noqa: E402
from qewton.geometries.continuous.domains_2d.rectangle import Rectangle  # noqa: E402
from qewton.graphs.control_nodes.graph_node import GraphNode  # noqa: E402
from qewton.graphs.graphs import Graph  # noqa: E402
from qewton.graphs.pipelines import PINNPipeline  # noqa: E402

LOGGER = logging.getLogger(__name__)
X, U = Variable("x", 2), Variable("u", 1)
DOMAIN = Rectangle(X, [0.0, 0.0], 1.0, 1.0)


def _wrapped(depth: int) -> GraphNode | FCN:
    model = FCN(in_neurons=X, hidden_neurons=4, out_neurons=U, n_hidden_layers=3)
    for _ in range(depth):
        graph = Graph()
        graph.add_node(model)
        model = GraphNode(graph, input_ports=model.input_ports, output_ports=model.output_ports)
    return model


def _seconds_per_call(function, repeat: int) -> float:
    function()
    start = time.perf_counter()
    for _ in range(repeat):
        function()
    return (time.perf_counter() - start) / repeat


def _nested(depth: int) -> tuple[float, float]:
    def build():
        graph = Graph()
        graph.connect(RandomUniformSampler(DOMAIN, n_points=4), _wrapped(depth))
        graph.setup()
        return graph

    return _seconds_per_call(build, 3), _seconds_per_call(build().run, 300)


def _pinn() -> tuple[float, float]:
    model = FCN(in_neurons=X, hidden_neurons=8, out_neurons=U, n_hidden_layers=3)

    def residual(u: U, x: X):  # type: ignore
        return u.laplacian(x) - 1.0

    def build():
        pipeline = PINNPipeline(
            RandomUniformSampler(DOMAIN, n_points=16), [model], residual=residual
        )
        pipeline.setup()
        return pipeline

    return _seconds_per_call(build, 3), _seconds_per_call(build().run, 100)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = [("PINN pipeline (FCN 3x8, laplacian)", *_pinn())]
    rows += [(f"FCN 3x4 in {d} wrapping GraphNodes", *_nested(d)) for d in (0, 2, 4)]
    for name, build, run in rows:
        LOGGER.info("%-40s build %9.1f ms   run %9.1f us", name, build * 1e3, run * 1e6)


if __name__ == "__main__":
    main()
