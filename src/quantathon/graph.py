"""Construcción del grafo de la red de transmisión a partir de los datos del SEN."""

import networkx as nx
import pandas as pd


def build_region_graph(db: pd.DataFrame, nodes: list[str]) -> nx.Graph:
    """Construye el subgrafo de la red restringido a un conjunto de nodos.

    Cada fila de ``db`` describe un circuito con el formato "Origen-Destino"
    en la columna ``Circuito``; se agrega una arista cuando ambos extremos
    pertenecen a ``nodes``, con peso igual a ``Shape__Length``.
    """
    node_set = set(nodes)
    G = nx.Graph()

    for _, fila in db.iterrows():
        circuito = str(fila["Circuito"]).strip()
        if "-" not in circuito:
            continue

        u, v = (parte.strip() for parte in circuito.split("-", 1))
        if u in node_set and v in node_set:
            G.add_edge(u, v, weight=fila["Shape__Length"])

    return G
