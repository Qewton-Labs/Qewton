"""Embedded states of a Plotly figure as the changes between them: a
snapshot is every drawn trace and the axes as flat ``{path: value}``
properties; frames, slider steps and dropdown buttons write only what
differs between the snapshots of their states."""

from plotly.utils import PlotlyJSONEncoder


def flatten(properties: dict, prefix: str = "") -> dict:
    """Nested properties as ``{"marker.color": ...}``; the binding of a
    trace to its subplot is left out, since it never changes.

    Args:
        properties: Plotly properties, e.g. of ``to_plotly_json()``.
        prefix: Path of `properties`, e.g. ``"xaxis"``.

    Returns:
        dict: ``{path: value}``.
    """
    flat = {}
    for key, value in properties.items():
        if not prefix and key in ("type", "uid", "xaxis", "yaxis", "scene"):
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def nest(flat: dict) -> dict:
    """The inverse of :func:`flatten`.

    Args:
        flat: ``{path: value}``.

    Returns:
        dict: Nested properties.
    """
    nested: dict = {}
    for path, value in flat.items():
        *parents, leaf = path.split(".")
        node = nested
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = value
    return nested


def _same(a, b) -> bool:
    encode = PlotlyJSONEncoder().encode
    return encode(a) == encode(b)


def _varying(snapshots: list[dict]) -> list[str]:
    # The keys whose values differ between the flat snapshots.
    keys = list(dict.fromkeys(k for s in snapshots for k in s))
    first = snapshots[0]
    return [
        k for k in keys if any(not _same(s.get(k), first.get(k)) for s in snapshots[1:])
    ]


def changes(snapshots) -> tuple[list[int], dict[int, list[str]], list[str]]:
    """What differs between the snapshots of the states of a control.

    Args:
        snapshots (list of tuple): Per state, ``({trace index: flat
            properties}, flat axis layout)``.

    Returns:
        tuple: The traces with properties differing between states, those
            properties per trace, and the differing axis layout keys.
    """
    trace_keys = {
        i: _varying([traces[i] for traces, _ in snapshots]) for i in snapshots[0][0]
    }
    indices = [i for i, keys in trace_keys.items() if keys]
    layout_keys = _varying([axes for _, axes in snapshots])
    return indices, {i: trace_keys[i] for i in indices}, layout_keys


def update_args(snapshots) -> list[list]:
    """The arguments of a ``method="update"`` call per state, setting the
    properties that differ between the states (one value per changed trace
    and key).

    Args:
        snapshots: As for :func:`changes`.

    Returns:
        list of list: ``[restyle, relayout, trace indices]`` per state.
    """
    indices, keys, layout_keys = changes(snapshots)
    union = list(dict.fromkeys(k for i in indices for k in keys[i]))
    return [
        [
            {k: [traces[i].get(k) for i in indices] for k in union},
            {k: axes.get(k) for k in layout_keys},
            indices,
        ]
        for traces, axes in snapshots
    ]
