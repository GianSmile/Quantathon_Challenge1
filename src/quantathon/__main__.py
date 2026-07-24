"""Punto de entrada: compara Max-Cut (fuerza bruta, greedy, GW y QAOA) sobre la red real."""

import pandas as pd

from quantathon.classical import cut_value, maxcut_brute_force, maxcut_greedy
from quantathon.config import REGION_CENTRAL_PACIFICO, ROOT_DIR
from quantathon.data import load_lineas_transmision
from quantathon.goemans_williamson import maxcut_goemans_williamson
from quantathon.graph import build_region_graph
from quantathon.qaoa import expected_cut_value, optimize_qaoa, run_qaoa
from quantathon.qubo import weight_matrix

MATRIZ_PESOS_CSV = ROOT_DIR / "matriz_pesos.csv"
MATRIZ_PESOS_HTML = ROOT_DIR / "matriz_pesos.html"

QAOA_P = 2
QAOA_SHOTS = 300
QAOA_MAXITER = 20


def main() -> None:
    db = load_lineas_transmision()
    G = build_region_graph(db, REGION_CENTRAL_PACIFICO)
    nodes = list(G.nodes())

    mejor_valor, _ = maxcut_brute_force(G, nodes)
    print(f"Nodos: {len(nodes)}")
    print(f"Óptimo (fuerza bruta): {mejor_valor:.1f}")

    valor_greedy = cut_value(G, maxcut_greedy(G, nodes), nodes)
    print(
        f"Greedy:                {valor_greedy:.1f} ({100 * valor_greedy / mejor_valor:.1f}% del óptimo)"
    )

    _, valor_gw = maxcut_goemans_williamson(G, nodes)
    print(f"Goemans-Williamson:    {valor_gw:.1f} ({100 * valor_gw / mejor_valor:.1f}% del óptimo)")

    print(f"Optimizando QAOA (p={QAOA_P})...")
    gammas, betas, _ = optimize_qaoa(
        G,
        p=QAOA_P,
        valor_referencia=mejor_valor,
        shots=QAOA_SHOTS,
        maxiter=QAOA_MAXITER,
        verbose=False,
    )
    conteos, nodes_qaoa = run_qaoa(G, gammas, betas, shots=QAOA_SHOTS)
    valor_qaoa = expected_cut_value(G, conteos, nodes_qaoa)
    print(
        f"QAOA:                  {valor_qaoa:.1f} ({100 * valor_qaoa / mejor_valor:.1f}% del óptimo)"
    )

    W, nodes_w = weight_matrix(G)
    tabla = pd.DataFrame(W, index=nodes_w, columns=nodes_w)
    tabla.to_csv(MATRIZ_PESOS_CSV)
    tabla.round(1).to_html(MATRIZ_PESOS_HTML, border=1)
    print(f"Matriz de pesos guardada en {MATRIZ_PESOS_CSV} y {MATRIZ_PESOS_HTML}")


if __name__ == "__main__":
    main()
