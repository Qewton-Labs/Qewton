"""A picture of the coupling graph together with its data, sets and coordinates.

`coupling_diagram(task)` collects what to draw as plain data and `plot_coupling`
draws it with matplotlib. Five columns, left to right:

    equations | unknowns | data | sets | coordinates

- equations: one node per (normalized) constraint, standing for all its components;
- unknowns: the unknown components of the coupling graph;
- data: the fixed and varying functions occurring in the equations;
- sets: where the equations are posed and the unknowns are defined;
- coordinates: the coordinates the sets live in.

Edge kinds: `occurs` (equation to data function), `contains` (equation to unknown
component, `matched` if the matching pairs them), `posed` (equation to set), `on`
(unknown or data function to a set it is evaluated on) and `in` (set to coordinate).
"""

from __future__ import annotations

from dataclasses import dataclass

from qewton.equations.analysis.context import AnalysisContext
from qewton.equations.analysis.prepare.signatures import evaluation_sets
from qewton.equations.geometry.sets import Domain, SetNode, Slice
from qewton.equations.geometry.coordinates import coordinate_factors, root_of
from qewton.equations.model.task import Role, Task

COLUMNS = ("equations", "unknowns", "data", "sets", "coordinates")
Node = tuple[str, str]


@dataclass(frozen=True)
class Edge:
    """A link between neighbouring columns."""

    kind: str
    left: Node
    right: Node
    matched: bool = False


@dataclass(frozen=True)
class CouplingDiagram:
    """Nodes per column in drawing order, the edges, and the role of each data node."""

    columns: dict[str, tuple[str, ...]]
    edges: tuple[Edge, ...]
    roles: dict[Node, str]


def _data_nodes(task: Task, constraint, roles: dict[Node, str]) -> list[Node]:
    found = []
    for symbol in sorted(constraint.symbols(), key=lambda s: str(s.variable.name)):
        if task.roles[symbol] is not Role.UNKNOWN:
            node = ("data", str(symbol.variable.name))
            roles[node] = task.roles[symbol].value
            found.append(node)
    return found


def _set_coordinates(domain: SetNode) -> list[str]:
    """The coordinates a set lives in, in the order of its factors."""
    if isinstance(domain, Slice):
        found = [root_of(domain.axis).name]
    elif isinstance(domain, Domain):
        found = [root_of(axis).name for axis in coordinate_factors(domain.ambient)]
    else:
        found = [name for part in domain.built_from() for name in _set_coordinates(part)]
    return list(dict.fromkeys(str(f) for f in found))


def _evaluated_on(ctx: AnalysisContext, data: list[str]):
    """(unknown component or data node, set) for every set a function is evaluated on."""
    graph = ctx.coupling
    evaluated = evaluation_sets(ctx.problem)
    for symbol, sets in evaluated.items():
        nodes: list[Node] = [
            ("unknowns", str(u)) for u in graph.unknowns if u.symbol == symbol
        ] or [("data", str(symbol.variable.name))]
        if nodes[0][0] == "data" and nodes[0][1] not in data:
            continue
        for domain in sorted(sets, key=str):
            for node in nodes:
                yield node, domain


def coupling_diagram(task: Task) -> CouplingDiagram:
    """The nodes and edges of the extended coupling graph of a task."""
    ctx = AnalysisContext(task)
    columns: dict[str, list[str]] = {name: [] for name in COLUMNS}
    edges: list[Edge] = []
    roles: dict[Node, str] = {}
    columns["unknowns"] = [str(unknown) for unknown in ctx.coupling.unknowns]

    domains: dict[str, SetNode] = {}

    def add(kind: str, left: Node, right: Node, matched: bool = False) -> None:
        for name, label in (left, right):
            if label not in columns[name]:
                columns[name].append(label)
        edge = Edge(kind, left, right, matched)
        if edge not in edges:
            edges.append(edge)

    for index, constraint in enumerate(ctx.problem.constraints):
        columns["equations"].append(f"c{index + 1}")
        for node in _data_nodes(task, constraint, roles):
            add("occurs", ("equations", f"c{index + 1}"), node)
        for domain in constraint.sets():
            domains[str(domain)] = domain
            add("posed", ("sets", str(domain)), ("equations", f"c{index + 1}"))
    for equation, unknown in ctx.coupling.edges:
        add(
            "contains",
            ("equations", f"c{equation.constraint + 1}"),
            ("unknowns", str(unknown)),
            (equation, unknown) in ctx.coupling.matching,
        )
    for unknown in ctx.coupling.unknowns:
        roles[("unknowns", str(unknown))] = "unknown"
    for node, domain in _evaluated_on(ctx, columns["data"]):
        domains[str(domain)] = domain
        add("on", node, ("sets", str(domain)))
    for label in columns["sets"]:
        for axis in _set_coordinates(domains[label]):
            add("in", ("sets", label), ("coordinates", axis))
    return CouplingDiagram(
        {name: tuple(nodes) for name, nodes in columns.items()}, tuple(edges), roles
    )


_COLUMN_STYLE = {
    "sets": ("tab:red", "sets"),
    "equations": ("tab:orange", "equations"),
    "unknowns": ("tab:green", "unknowns"),
    "data": ("tab:purple", "data"),
    "coordinates": ("tab:gray", "coordinates"),
}
_ROLE_COLOR = {"unknown": "tab:green", "fixed": "tab:purple", "varying": "tab:pink"}
_EDGE_STYLE = {
    "occurs": ("0.55", "--", 1.0),
    "posed": ("tab:red", ":", 1.3),
    "on": ("tab:green", "-", 0.8),
    "in": ("tab:red", "-", 0.8),
    "depends": ("0.7", "-", 1.0),
    "contains": ("0.75", "-", 1.0),
}


def _positions(diagram: CouplingDiagram) -> dict[Node, tuple[float, float]]:
    position = {}
    for x, name in enumerate(COLUMNS):
        count = len(diagram.columns[name])
        for i, label in enumerate(diagram.columns[name]):
            position[(name, label)] = (x, 1 - (i + 0.5) / count)
    return position


def plot_coupling(task: Task, ax=None):
    """Draw `coupling_diagram(task)` and return the matplotlib axes."""
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    diagram = coupling_diagram(task)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    position = _positions(diagram)
    _draw_edges(ax, diagram, position)
    _draw_nodes(ax, diagram, position)
    for x, name in enumerate(COLUMNS):
        ax.text(x, 1.06, _COLUMN_STYLE[name][1], ha="center", weight="bold")
    _draw_legend(ax, Line2D)
    ax.set_xlim(-0.5, len(COLUMNS) - 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.axis("off")
    return ax


def _draw_edges(ax, diagram, position) -> None:
    for edge in diagram.edges:
        (x0, y0), (x1, y1) = position[edge.left], position[edge.right]
        color, style, width = _EDGE_STYLE[edge.kind]
        if edge.matched:
            color, width = "tab:blue", 2.8
        ax.plot([x0, x1], [y0, y1], color=color, ls=style, lw=width, zorder=1)


def _draw_nodes(ax, diagram, position) -> None:
    for (name, label), (x, y) in position.items():
        role = diagram.roles.get((name, label))
        ax.text(
            x,
            y,
            f"{label}\n[{role}]" if role else label,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.4",
                "fc": _ROLE_COLOR.get(role, _COLUMN_STYLE[name][0]),
                "ec": "none",
            },
        )


def _draw_legend(ax, line_class) -> None:
    ax.legend(
        handles=[
            line_class([0], [0], color="tab:blue", lw=2.8, label="matching"),
            line_class([0], [0], color="0.75", label="contains unknown"),
            line_class([0], [0], color="0.55", ls="--", label="contains data"),
            line_class([0], [0], color="tab:red", ls=":", label="equation posed on"),
            line_class([0], [0], color="tab:green", lw=0.8, label="unknown defined on"),
            line_class([0], [0], color="tab:red", lw=0.8, label="set lives in"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
    )
