from graphviz import Digraph
from qewton.graphs.graphs import Graph


class GraphPlotter:
    def __init__(self, graph: Graph):
        self.graph = graph

    def to_dot(self) -> Digraph:
        dot = Digraph("QewtonGraph")
        # assign unique IDs
        node_ids = {node: str(i) for i, node in enumerate(self.graph.nodes)}

        # General appearance
        dot.attr(rankdir="LR")  # Left -> Right flow
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

            label = f"{node.name}\n" f"({type(node).__name__})"
            dot.node(node_ids[node], label=label)

        # --------------------
        # Create edges
        # --------------------
        for node in self.graph.nodes:

            for edge in self.graph.outgoing_edges[node]:

                source_node = edge.from_port.node
                target_node = edge.to_port.node

                source_label = edge.from_port.name
                target_label = edge.to_port.name

                dot.edge(
                    node_ids[source_node],
                    node_ids[target_node],
                    # so far no labels on edges
                    # label=f"{source_label} → {target_label}",
                )

        return dot

    def save_png(self, filename: str = "graph"):
        dot = self.to_dot()
        dot.render(filename, format="png", cleanup=True)

    def save_pdf(self, filename: str = "graph"):
        dot = self.to_dot()
        dot.render(filename, format="pdf", cleanup=True)

    def save_svg(self, filename: str = "graph"):
        dot = self.to_dot()
        dot.render(filename, format="svg", cleanup=True)

    # def show(self):
    #    dot = self.to_dot()
    #    dot.view()
    # No such file or directory: 'xdg-open'
