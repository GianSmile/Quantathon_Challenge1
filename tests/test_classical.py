import networkx as nx

from quantathon.classical import cut_value, maxcut_brute_force, maxcut_greedy


def test_cut_value_counts_only_crossing_edges():
    G = nx.Graph()
    G.add_edge("A", "B", weight=2.0)
    G.add_edge("B", "C", weight=3.0)
    nodes = ["A", "B", "C"]

    assert cut_value(G, "010", nodes) == 5.0
    assert cut_value(G, "000", nodes) == 0.0


def test_maxcut_brute_force_triangle():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "A", weight=1.0)
    nodes = ["A", "B", "C"]

    mejor_valor, mejores_bitstrings = maxcut_brute_force(G, nodes)

    assert mejor_valor == 2.0
    assert len(mejores_bitstrings) == 6
    for bitstring in mejores_bitstrings:
        assert cut_value(G, bitstring, nodes) == mejor_valor


def test_maxcut_brute_force_empty_graph():
    G = nx.Graph()

    mejor_valor, mejores_bitstrings = maxcut_brute_force(G, nodes=[])

    assert mejor_valor == 0.0
    assert mejores_bitstrings == [""]


def test_maxcut_greedy_matches_optimal_on_bipartite_path():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "D", weight=1.0)
    nodes = ["A", "B", "C", "D"]

    bitstring = maxcut_greedy(G, nodes)
    mejor_valor, _ = maxcut_brute_force(G, nodes)

    assert cut_value(G, bitstring, nodes) == mejor_valor


def test_maxcut_greedy_prefers_side_with_more_assigned_weight():
    G = nx.Graph()
    G.add_edge("A", "C", weight=20.0)
    G.add_edge("B", "C", weight=5.0)
    nodes = ["A", "B", "C"]

    bitstring = maxcut_greedy(G, nodes)
    asignacion = dict(zip(nodes, bitstring))

    # A y B quedan del mismo lado (0); C debe cruzar hacia el lado opuesto
    # porque su arista más pesada (hacia A) pesa más que la liviana hacia B.
    assert asignacion["A"] == asignacion["B"]
    assert asignacion["C"] != asignacion["A"]


def test_maxcut_greedy_empty_graph():
    G = nx.Graph()

    assert maxcut_greedy(G, nodes=[]) == ""
