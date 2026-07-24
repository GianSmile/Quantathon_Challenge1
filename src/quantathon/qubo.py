"""Formulación QUBO de Max-Cut sobre la red de transmisión."""

import networkx as nx
import numpy as np


def weight_matrix(G: nx.Graph) -> tuple[np.ndarray, list]:
    """Convierte un grafo con nodos nombrados en una matriz de pesos indexada.

    Devuelve ``(W, nodes)`` donde ``nodes[i]`` es el nombre del nodo en la
    fila/columna ``i`` de ``W``, y ``W[i, j]`` es el peso de la arista entre
    ``nodes[i]`` y ``nodes[j]`` (0 si no existe esa arista).
    """
    nodes = list(G.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))

    for u, v, w in G.edges(data="weight"):
        i, j = index[u], index[v]
        W[i, j] = w
        W[j, i] = w

    return W, nodes
