"""Soluciones clásicas de Max-Cut: fuerza bruta (óptimo) y greedy (heurística)."""

import itertools

import networkx as nx


def cut_value(G: nx.Graph, bitstring: str, nodes: list[str]) -> float:
    """Calcula el valor del corte inducido por ``bitstring`` sobre ``nodes``.

    ``bitstring[i]`` indica a qué lado del corte queda ``nodes[i]``; el valor
    devuelto es la suma de los pesos de las aristas cuyos extremos caen en
    lados distintos.
    """
    asignacion = dict(zip(nodes, map(int, bitstring)))

    return sum(
        float(data.get("weight", 1.0))
        for u, v, data in G.edges(data=True)
        if asignacion[u] != asignacion[v]
    )


def maxcut_brute_force(G: nx.Graph, nodes: list[str]) -> tuple[float, list[str]]:
    """Resuelve Max-Cut probando exhaustivamente todas las asignaciones.

    Devuelve ``(mejor_valor, mejores_bitstrings)``: el valor óptimo del corte
    y todas las asignaciones (bitstrings) que lo alcanzan. Complejidad
    exponencial en ``len(nodes)``; solo viable como referencia en grafos
    pequeños.
    """
    mejor_valor = -1.0
    mejores_bitstrings = []

    for bits in itertools.product("01", repeat=len(nodes)):
        bitstring = "".join(bits)
        valor = cut_value(G, bitstring, nodes)

        if valor > mejor_valor:
            mejor_valor = valor
            mejores_bitstrings = [bitstring]
        elif valor == mejor_valor:
            mejores_bitstrings.append(bitstring)

    return mejor_valor, mejores_bitstrings


def maxcut_greedy(G: nx.Graph, nodes: list[str]) -> str:
    """Aproxima Max-Cut asignando cada nodo, en el orden de ``nodes``, al lado
    que maximiza el peso cruzado con los vecinos que ya tienen lado asignado.

    Es una heurística local (no vuelve atrás sobre decisiones previas): corre
    en tiempo polinomial, a diferencia de ``maxcut_brute_force``, pero no
    garantiza el óptimo.
    """
    lado: dict[str, int] = {}

    for nodo in nodes:
        peso_hacia_0 = 0.0
        peso_hacia_1 = 0.0

        for vecino, data in G[nodo].items():
            if vecino not in lado:
                continue

            peso = float(data.get("weight", 1.0))
            if lado[vecino] == 0:
                peso_hacia_0 += peso
            else:
                peso_hacia_1 += peso

        # Asignar al lado opuesto de donde ya pesa más maximiza el corte local.
        lado[nodo] = 0 if peso_hacia_1 >= peso_hacia_0 else 1

    return "".join(str(lado[nodo]) for nodo in nodes)
