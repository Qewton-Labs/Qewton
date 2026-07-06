from graphviz import Digraph
from qewton.graphs.graphs import Graph


class GraphPlotter:
    def __init__(self, graph: Graph):
        self.graph = graph

    def _node_style(self, node):
        COLORS = {
            "FCN": "lightskyblue",
            "MSEConstraint": "aquamarine",
            "DataLoader": "lemonchiffon",
        }
        return {"fillcolor": COLORS.get(type(node).__name__, "lightblue")}

    # ----------------------------
    # vertical port stack
    # ----------------------------
    def _stack_ports(self, ports, prefix):
        return "{ " + " | ".join(f"<{prefix}_{p.name}> {p.name}" for p in ports) + " }"

    def to_dot(self) -> Digraph:
        dot = Digraph("QewtonGraph")

        node_ids = {node: str(i) for i, node in enumerate(self.graph.nodes)}

        dot.attr(rankdir="LR")
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor="paleturquoise",
            fontname="Arial",
        )
        dot.attr("edge", arrowsize="0.8")

        # --------------------
        # Create nodes
        # --------------------
        for node in self.graph.nodes:
            style = self._node_style(node)

            if type(node).__name__ == "FCN":
                # --------------------
                # inner structure of FCN
                # --------------------
                fcn_graph = getattr(node, "_graph", None)
                inner_nodes = list(getattr(fcn_graph, "sorted_nodes", []) or [])

                inputs = list(getattr(node, "input_ports", []) or [])
                outputs = list(getattr(node, "output_ports", []) or [])

                node_id = node_ids[node]

                with dot.subgraph(name=f"cluster_{node_id}") as c:  # type: ignore
                    c.attr(label=f"FCN ({len(inner_nodes)} layers)")
                    c.attr(style="rounded")
                    c.attr(color="darkblue")
                    c.attr(compound="true")

                    # -------------------------
                    # interface nodes inside cluster
                    # -------------------------
                    in_id = f"{node_id}_in"
                    out_id = f"{node_id}_out"

                    if inputs:
                        c.node(
                            in_id,
                            label=self._stack_ports(inputs, "in"),
                            shape="record",
                            style="rounded,filled",
                            fillcolor=self._node_style(node)["fillcolor"],
                        )

                    if outputs:
                        c.node(
                            out_id,
                            label=self._stack_ports(outputs, "out"),
                            shape="record",
                            style="rounded,filled",
                            fillcolor=self._node_style(node)["fillcolor"],
                        )

                    # -------------------------
                    # inner layers
                    # -------------------------
                    layer_ids = []

                    for idx, layer in enumerate(inner_nodes):
                        layer_id = f"{node_id}_layer_{idx}"

                        label = type(layer).__name__
                        if hasattr(layer, "in_neurons"):
                            label += (
                                f"\n{layer.in_neurons.value} → {layer.out_neurons.value}"
                            )

                        c.node(layer_id, label)
                        layer_ids.append(layer_id)

                    for i in range(len(layer_ids) - 1):
                        c.edge(layer_ids[i], layer_ids[i + 1])

                    if layer_ids:
                        if inputs:
                            c.edge(in_id, layer_ids[0])
                        if outputs:
                            c.edge(layer_ids[-1], out_id)
            else:
                inputs = list(getattr(node, "input_ports", []) or [])
                outputs = list(getattr(node, "output_ports", []) or [])

                has_inputs = len(inputs) > 0
                has_outputs = len(outputs) > 0

                if has_inputs or has_outputs:
                    left = self._stack_ports(inputs, "in") if has_inputs else ""
                    right = self._stack_ports(outputs, "out") if has_outputs else ""

                    if has_inputs and has_outputs:
                        label = f"{{ {left} | {node.name} | {right} }}"
                    elif has_inputs:
                        label = f"{{ {left} | {node.name} }}"
                    else:
                        label = f"{{ {node.name} | {right} }}"

                    dot.node(
                        node_ids[node],
                        label=label,
                        shape="record",
                        style="rounded,filled",
                        fillcolor=style["fillcolor"],
                    )
                else:
                    dot.node(node_ids[node], label=node.name, **style)
        # --------------------
        # Create edges
        # --------------------
        for node in self.graph.nodes:
            for edge in self.graph.outgoing_edges[node]:

                source_node = edge.from_port.node
                target_node = edge.to_port.node

                source_id = node_ids[source_node]
                target_id = node_ids[target_node]

                source_conf = self.graph.dynamic_data_configs[source_node][edge.from_port]
                source_conf_str = f"[{', '.join(str(a) for a in source_conf.axes)}]"

                if type(source_node).__name__ == "FCN":
                    source_id = f"{source_id}_out"

                if type(target_node).__name__ == "FCN":
                    target_id = f"{target_id}_in"

                dot.edge(
                    source_id,
                    target_id,
                    label=source_conf_str,
                )

        return dot

    def save_png(self, filename: str = "graph"):
        self.to_dot().render(filename, format="png", cleanup=True)

    def save_pdf(self, filename: str = "graph"):
        self.to_dot().render(filename, format="pdf", cleanup=True)

    def save_svg(self, filename: str = "graph"):
        self.to_dot().render(filename, format="svg", cleanup=True)
