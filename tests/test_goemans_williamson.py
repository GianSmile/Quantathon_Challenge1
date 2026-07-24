import networkx as nx
import numpy as np

from quantathon.classical import cut_value, maxcut_brute_force
from quantathon.goemans_williamson import maxcut_goemans_williamson, solve_sdp


def test_solve_sdp_triangle_matches_known_symmetric_solution():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "A", weight=1.0)
    nodes = ["A", "B", "C"]

    X = solve_sdp(G, nodes)

    assert np.allclose(np.diag(X), 1.0, atol=1e-4)
    off_diagonal = X[~np.eye(3, dtype=bool)]
    assert np.allclose(off_diagonal, -0.5, atol=1e-3)


def test_maxcut_goemans_williamson_finds_optimum_on_triangle():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "A", weight=1.0)
    nodes = ["A", "B", "C"]

    bitstring, valor = maxcut_goemans_williamson(
        G, nodes, intentos=20, rng=np.random.default_rng(0)
    )
    mejor_valor, _ = maxcut_brute_force(G, nodes)

    assert valor == mejor_valor
    assert cut_value(G, bitstring, nodes) == valor


def test_maxcut_goemans_williamson_reaches_near_optimal_on_weighted_graph():
    G = nx.Graph()
    G.add_edge("A", "B", weight=10.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "D", weight=8.0)
    G.add_edge("D", "A", weight=1.0)
    nodes = ["A", "B", "C", "D"]

    bitstring, valor = maxcut_goemans_williamson(
        G, nodes, intentos=20, rng=np.random.default_rng(0)
    )
    mejor_valor, _ = maxcut_brute_force(G, nodes)

    assert cut_value(G, bitstring, nodes) == valor
    assert valor >= 0.878 * mejor_valor
