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
            fillcolor="lightblue",
            fontname="Arial",
        )
        dot.attr("edge", arrowsize="0.8")

        # --------------------
        # Create nodes
        # --------------------
        for node in self.graph.nodes:
            style = self._node_style(node)

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
            # if no ports (?)
            else:
                dot.node(node_ids[node], label=node.name, **style)

        # --------------------
        # Create edges
        # --------------------
        for node in self.graph.nodes:
            for edge in self.graph.outgoing_edges[node]:

                source_node = edge.from_port.node
                target_node = edge.to_port.node

                source_conf = self.graph.dynamic_data_configs[source_node][edge.from_port]
                source_conf_str = f"[{', '.join(str(a) for a in source_conf.axes)}]"

                dot.edge(
                    f"{node_ids[source_node]}:out_{edge.from_port.name}",
                    f"{node_ids[target_node]}:in_{edge.to_port.name}",
                    label=source_conf_str,
                )

        return dot

    def save_png(self, filename: str = "graph"):
        self.to_dot().render(filename, format="png", cleanup=True)

    def save_pdf(self, filename: str = "graph"):
        self.to_dot().render(filename, format="pdf", cleanup=True)

    def save_svg(self, filename: str = "graph"):
        self.to_dot().render(filename, format="svg", cleanup=True)
