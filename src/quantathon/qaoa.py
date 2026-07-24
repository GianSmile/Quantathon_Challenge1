"""QAOA para Max-Cut: circuito Guppy, ejecución en emulador y ajuste de ángulos."""

import math
from collections import Counter

import networkx as nx
import numpy as np
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, owned, result
from guppylang.std.quantum import cx, h, measure_array, qubit, rx, rz
from scipy.optimize import minimize

from quantathon.classical import cut_value

DEFAULT_SHOTS = 1000
DEFAULT_SEED = 12345


def build_qaoa_program(G: nx.Graph, gammas: list[float], betas: list[float]):
    """Construye el programa Guppy de QAOA (``p = len(gammas)`` capas) para ``G``.

    Los ángulos y las aristas (con pesos normalizados por el peso máximo, para
    evitar rotaciones demasiado grandes) quedan capturados como constantes de
    Python al construir el circuito; el programa Guppy resultante solo recibe
    los qubits. Devuelve ``(main, nodes)``: el programa Guppy ejecutable y el
    orden de nodos que corresponde a cada bit medido.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    gammas = [float(gamma) for gamma in gammas]
    betas = [float(beta) for beta in betas]

    if n == 0:
        raise ValueError("El grafo no puede estar vacío.")

    if G.number_of_edges() == 0:
        raise ValueError("El grafo debe tener al menos una arista.")

    if len(gammas) != len(betas):
        raise ValueError("Debe haber la misma cantidad de gammas y betas.")

    if len(gammas) == 0:
        raise ValueError("P debe ser al menos 1.")

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    max_weight = max(abs(float(data.get("weight", 1.0))) for _, _, data in G.edges(data=True))
    if max_weight == 0:
        raise ValueError("Al menos una arista debe tener peso distinto de cero.")

    edges = [
        (node_to_idx[u], node_to_idx[v], float(data.get("weight", 1.0)) / max_weight)
        for u, v, data in G.edges(data=True)
    ]

    # Cada elemento representa una capa QAOA: (gamma_l, beta_l).
    layers = list(zip(gammas, betas))

    @guppy
    def qaoa(q: array[qubit, comptime(n)] @ owned) -> array[qubit, comptime(n)]:
        # Estado inicial |+>^n.
        for i in range(comptime(n)):
            h(q[i])

        # Repetir p capas costo-mezclador.
        for gamma_layer, beta_layer in comptime(layers):
            # Hamiltoniano de costo.
            for u, v, weight in comptime(edges):
                cx(q[u], q[v])
                rz(q[v], angle(gamma_layer * weight / comptime(math.pi)))
                cx(q[u], q[v])

            # Hamiltoniano mezclador.
            for i in range(comptime(n)):
                rx(q[i], angle(2.0 * beta_layer / comptime(math.pi)))

        return q

    @guppy
    def main() -> None:
        q = array(qubit() for _ in range(comptime(n)))
        q = qaoa(q)
        bits = measure_array(q)
        result("c", bits)

    return main, nodes


def run_qaoa(
    G: nx.Graph,
    gammas: list[float],
    betas: list[float],
    shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
) -> tuple[Counter[str], list[str]]:
    """Compila y simula el circuito QAOA de ``G`` con los ángulos dados.

    Devuelve ``(conteos, nodes)``: cuántas veces salió cada bitstring en
    ``shots`` mediciones, y el orden de nodos correspondiente a cada bit.
    """
    main, nodes = build_qaoa_program(G, gammas, betas)

    resultado = (
        main.emulator(n_qubits=len(nodes)).statevector_sim().with_shots(shots).with_seed(seed).run()
    )

    return resultado.register_counts()["c"], nodes


def expected_cut_value(G: nx.Graph, counts: Counter[str], nodes: list[str]) -> float:
    """Valor esperado del corte a partir de conteos de bitstrings medidos.

    Siempre se evalúa con los pesos originales de ``G`` (no los normalizados
    que usa el circuito internamente), vía ``cut_value``.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        raise ValueError("No se obtuvieron mediciones.")

    return (
        sum(frecuencia * cut_value(G, bitstring, nodes) for bitstring, frecuencia in counts.items())
        / total_shots
    )


def optimize_qaoa(
    G: nx.Graph,
    p: int = 2,
    valor_referencia: float | None = None,
    shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
    gamma_inicial: float = 1.330995,
    beta_inicial: float = 1.219451,
    maxiter: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Ajusta ``(gammas, betas)`` con COBYLA para maximizar el corte esperado.

    Parte de ``gamma_inicial``/``beta_inicial`` repetidos en las ``p`` capas
    (valores por defecto: óptimo conocido para ``p=1``). ``valor_referencia``
    (p.ej. el óptimo de fuerza bruta) es opcional: si se da, cada evaluación
    registra además la razón de aproximación alcanzada hasta el momento.

    Devuelve ``(gammas_opt, betas_opt, historial)``, donde ``historial`` es
    la lista de evaluaciones de COBYLA en orden, útil para graficar la
    convergencia.
    """
    maxiter = maxiter if maxiter is not None else 60 * p
    parametros_iniciales = np.concatenate([np.full(p, gamma_inicial), np.full(p, beta_inicial)])
    historial: list[dict] = []

    def objetivo(parametros: np.ndarray) -> float:
        gammas, betas = parametros[:p], parametros[p:]
        conteos, nodes = run_qaoa(G, gammas, betas, shots=shots, seed=seed)
        valor_esperado = expected_cut_value(G, conteos, nodes)

        registro = {
            "evaluacion": len(historial) + 1,
            "gammas": gammas.tolist(),
            "betas": betas.tolist(),
            "valor_esperado": valor_esperado,
        }
        if valor_referencia is not None:
            registro["razon_aproximacion"] = valor_esperado / valor_referencia
        historial.append(registro)

        if verbose:
            extra = (
                f" | razón={100 * registro['razon_aproximacion']:.2f}%" if valor_referencia else ""
            )
            print(
                f"Evaluación {registro['evaluacion']:03d} | "
                f"gammas={np.round(gammas, 4)} | betas={np.round(betas, 4)} | "
                f"E[C]={valor_esperado:.2f}{extra}"
            )

        # COBYLA minimiza.
        return -valor_esperado

    resultado = minimize(
        objetivo, parametros_iniciales, method="COBYLA", options={"maxiter": maxiter}
    )
    gammas_opt, betas_opt = resultado.x[:p], resultado.x[p:]

    return gammas_opt, betas_opt, historial
