import networkx as nx
import numpy as np
import pytest

from quantathon.nexus import (
    bitstring_from_pytket_key,
    build_qaoa_circuit,
    counts_from_pytket_results,
    results_table,
)


def _triangulo():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "A", weight=1.0)
    return G, ["A", "B", "C"]


def test_build_qaoa_circuit_shape():
    G, _ = _triangulo()

    circuito, nodes = build_qaoa_circuit(G, gammas=[0.8, 0.3], betas=[0.4, 0.6])

    assert nodes == ["A", "B", "C"]
    assert circuito.n_qubits == 3
    assert circuito.n_bits == 3


def test_build_qaoa_circuit_rejects_graph_without_edges():
    G = nx.Graph()
    G.add_node("A")

    with pytest.raises(ValueError, match="al menos una arista"):
        build_qaoa_circuit(G, gammas=[0.5], betas=[0.5])


@pytest.mark.parametrize(
    "clave,esperado",
    [
        ("010", "010"),
        ("0b010", "010"),
        ((0, 1, 0), "010"),
        ([0, 1, 0], "010"),
        (np.array([0, 1, 0]), "010"),
    ],
)
def test_bitstring_from_pytket_key_formats(clave, esperado):
    assert bitstring_from_pytket_key(clave, n_qubits=3) == esperado


def test_bitstring_from_pytket_key_wrong_length_raises():
    with pytest.raises(ValueError):
        bitstring_from_pytket_key("01", n_qubits=3)


def test_counts_from_pytket_results_aggregates_by_bitstring():
    conteos_crudos = {(0, 1, 0): 5, "010": 3, (1, 0, 0): 2}

    conteos = counts_from_pytket_results(conteos_crudos, n_qubits=3)

    assert conteos["010"] == 8
    assert conteos["100"] == 2
    assert sum(conteos.values()) == 10


def test_results_table_reports_cut_value_and_optimal_ratio():
    G, nodes = _triangulo()
    conteos = counts_from_pytket_results({"010": 6, "000": 4}, n_qubits=3)

    tabla = results_table(G, conteos, nodes, valor_optimo=2.0)

    fila_010 = tabla[tabla["bitstring"] == "010"].iloc[0]
    assert fila_010["valor del corte"] == 2.0
    assert fila_010["razón del óptimo (%)"] == 100.0
    assert fila_010["frecuencia"] == 6

    fila_000 = tabla[tabla["bitstring"] == "000"].iloc[0]
    assert fila_000["valor del corte"] == 0.0

    assert tabla.iloc[0]["bitstring"] == "010"  # ordenado por frecuencia descendente
