"""Visualización de grafos de la red de transmisión."""

import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(G: nx.Graph, title: str = "") -> None:
    """Dibuja un grafo con etiquetas de nodos y pesos de las aristas."""
    plt.figure(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, font_size=10)

    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels = {k: f"{v:.0f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.show()
