import networkx as nx
import numpy as np

from quantathon.qubo import weight_matrix


def test_weight_matrix_matches_graph_edges():
    G = nx.Graph()
    G.add_edge("San Miguel", "El Este", weight=20465.6)
    G.add_edge("San Miguel", "Coronado", weight=5908.8)
    G.add_edge("San Miguel", "Colima1", weight=5641.2)
    G.add_edge("El Este", "Coronado", weight=14522.7)

    W, nodes = weight_matrix(G)
    idx = {node: i for i, node in enumerate(nodes)}

    assert W.shape == (4, 4)
    assert np.array_equal(W, W.T)
    assert np.all(np.diag(W) == 0)

    assert W[idx["San Miguel"], idx["El Este"]] == 20465.6
    assert W[idx["San Miguel"], idx["Coronado"]] == 5908.8
    assert W[idx["San Miguel"], idx["Colima1"]] == 5641.2
    assert W[idx["El Este"], idx["Coronado"]] == 14522.7

    # No existe línea directa entre estos pares.
    assert W[idx["El Este"], idx["Colima1"]] == 0
    assert W[idx["Coronado"], idx["Colima1"]] == 0


def test_weight_matrix_empty_graph():
    W, nodes = weight_matrix(nx.Graph())

    assert nodes == []
    assert W.shape == (0, 0)
