"""Aproximación de Max-Cut vía Goemans-Williamson (relajación SDP + redondeo)."""

import cvxpy as cp
import networkx as nx
import numpy as np

from quantathon.classical import cut_value


def solve_sdp(G: nx.Graph, nodes: list[str]) -> np.ndarray:
    """Resuelve la relajación SDP de Max-Cut y devuelve la matriz de Gram ``X``.

    ``X[i, j]`` es el producto punto ``v_i · v_j`` de los vectores unitarios
    óptimos para los nodos ``nodes[i]`` y ``nodes[j]`` (``X[i, i] = 1``).
    Maximiza ``Σ peso_ij · (1 − X[i,j]) / 2``, sujeto a que ``X`` sea
    semidefinida positiva con diagonal 1.
    """
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    X = cp.Variable((n, n), PSD=True)
    corte_esperado = cp.sum(
        [
            float(data.get("weight", 1.0)) * (1 - X[index[u], index[v]]) / 2
            for u, v, data in G.edges(data=True)
        ]
    )

    problema = cp.Problem(cp.Maximize(corte_esperado), [cp.diag(X) == 1])
    problema.solve()

    return X.value


def vectors_from_gram(X: np.ndarray) -> np.ndarray:
    """Descompone una matriz de Gram ``X = V @ V.T`` y devuelve ``V``.

    Cada fila de ``V`` es el vector (no necesariamente en la misma dimensión
    original) asociado a un nodo. Los autovalores negativos que puedan surgir
    por errores numéricos del solver se recortan a 0.
    """
    X = (X + X.T) / 2
    autovalores, autovectores = np.linalg.eigh(X)
    autovalores = np.clip(autovalores, 0, None)

    return autovectores @ np.diag(np.sqrt(autovalores))


def maxcut_goemans_williamson(
    G: nx.Graph, nodes: list[str], intentos: int = 20, rng: np.random.Generator | None = None
) -> tuple[str, float]:
    """Aproxima Max-Cut con el algoritmo de Goemans-Williamson.

    Resuelve la relajación SDP una sola vez y luego prueba ``intentos``
    hiperplanos aleatorios distintos para redondear los vectores a una
    asignación binaria, quedándose con el mejor corte encontrado (evaluado
    con los pesos originales, vía ``cut_value``).

    Devuelve ``(mejor_bitstring, mejor_valor)``. En expectativa, el valor
    obtenido es al menos ~0.878 veces el óptimo exacto, para cualquier grafo
    con pesos no negativos.
    """
    rng = rng or np.random.default_rng()

    X = solve_sdp(G, nodes)
    V = vectors_from_gram(X)

    mejor_bitstring = ""
    mejor_valor = -1.0

    for _ in range(intentos):
        hiperplano = rng.normal(size=V.shape[1])
        lado = V @ hiperplano >= 0
        bitstring = "".join("1" if bit else "0" for bit in lado)
        valor = cut_value(G, bitstring, nodes)

        if valor > mejor_valor:
            mejor_valor = valor
            mejor_bitstring = bitstring

    return mejor_bitstring, mejor_valor
