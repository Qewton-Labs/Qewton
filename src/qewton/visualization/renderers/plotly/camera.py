"""Moving the cameras of linked 3D cells together.

Plotly has no linked cameras: a script in the page listens for camera
changes of a scene and passes them on to the scenes of its group. It runs
in HTML outputs (:func:`page_script`) and as a browser callback in Dash
(:func:`dash_callback`); Plotly's own notebook output runs no scripts.
"""

import json

# The scenes of `groups` take over the camera of any scene of their group
# that `changes` (a relayout event) moves.
_SYNC = """
function (gd, groups, changes) {
    if (!changes || gd.__qewtonSyncing) { return; }
    var update = {};
    groups.forEach(function (group) {
        group.forEach(function (scene) {
            var camera = changes[scene + '.camera'];
            if (!camera) { return; }
            group.forEach(function (other) {
                if (other !== scene) { update[other + '.camera'] = camera; }
            });
        });
    });
    if (Object.keys(update).length === 0) { return; }
    gd.__qewtonSyncing = true;
    Plotly.relayout(gd, update).then(function () { gd.__qewtonSyncing = false; });
}
"""


def page_script(groups: list[list[str]]) -> str:
    """A script for a page holding the figure (Plotly's ``post_script``,
    which replaces ``{plot_id}`` by the id of the figure's element).

    Args:
        groups: Scene names of the linked cells, e.g.
            ``[["scene", "scene2"]]``.

    Returns:
        str: JavaScript.
    """
    return (
        "var gd = document.getElementById('{plot_id}');"
        f"var sync = {_SYNC};"
        f"var groups = {json.dumps(groups)};"
        "gd.on('plotly_relayout', function (changes) { sync(gd, groups, changes); });"
    )


def dash_callback(groups: list[list[str]], graph_id: str) -> str:
    """A Dash clientside callback taking the ``relayoutData`` of the graph
    `graph_id` and returning ``no_update``.

    Args:
        groups: Scene names of the linked cells.
        graph_id: Id of the ``dcc.Graph``.

    Returns:
        str: JavaScript function.
    """
    return (
        "function (changes) {"
        f"  var gd = document.getElementById({json.dumps(graph_id)})"
        "      .querySelector('.js-plotly-plot');"
        f"  ({_SYNC})(gd, {json.dumps(groups)}, changes);"
        "  return window.dash_clientside.no_update;"
        "}"
    )
