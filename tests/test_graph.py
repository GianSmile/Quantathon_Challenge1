import pandas as pd

from quantathon.graph import build_region_graph


def test_build_region_graph_filters_nodes_outside_region():
    db = pd.DataFrame(
        {
            "Circuito": ["A-B", "B-C", "C-D"],
            "Shape__Length": [10.0, 20.0, 30.0],
        }
    )

    G = build_region_graph(db, nodes=["A", "B", "C"])

    assert set(G.nodes) == {"A", "B", "C"}
    assert G["A"]["B"]["weight"] == 10.0
    assert not G.has_edge("C", "D")


def test_build_region_graph_ignores_rows_without_separator():
    db = pd.DataFrame({"Circuito": ["SoloUnNodo"], "Shape__Length": [5.0]})

    G = build_region_graph(db, nodes=["SoloUnNodo"])

    assert G.number_of_edges() == 0
