"""Ejecución de QAOA en hardware/emulador de Quantinuum vía Nexus (qnexus + pytket).

Este módulo **no** es parte de las dependencias instaladas del proyecto: el
paquete ``qnexus`` fija ``pandas<3``, incompatible con el ``pandas>=3.0.3``
que usa el resto de quantathon (`uv add qnexus` falla por ese choque). Está
pensado para correr en un entorno aparte con acceso a Quantinuum Nexus
—típicamente el propio "Nexus Lab" de Quantinuum, que ya trae ``qnexus`` y
``pytket`` preinstalados—, no vía ``uv run quantathon``.

Para usarlo fuera de Nexus Lab, instalá en ese entorno aparte::

    pip install qnexus pytket

y autenticate antes de llamar a ``run_qaoa_on_nexus``::

    import qnexus as qnx
    # Fuera de Nexus Lab, descomentar la línea siguiente:
    # qnx.login()
"""

import math
import uuid
from collections import Counter
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from pytket import Circuit

from quantathon.classical import cut_value

try:
    import qnexus as qnx
except ImportError:
    qnx = None


def build_qaoa_circuit(
    G: nx.Graph, gammas: list[float], betas: list[float]
) -> tuple[Circuit, list[str]]:
    """Construye el circuito QAOA de Max-Cut como un ``pytket.Circuit``.

    Misma lógica que ``qaoa.build_qaoa_program`` (estado inicial ``|+>^n``,
    capas costo-mezclador con pesos normalizados por el peso máximo), pero en
    Pytket en vez de Guppy: es el formato que requieren los backends de
    Quantinuum vía Nexus. Devuelve ``(circuito, nodes)``.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    gammas = [float(gamma) for gamma in gammas]
    betas = [float(beta) for beta in betas]

    if G.number_of_edges() == 0:
        raise ValueError("El grafo debe tener al menos una arista.")

    node_to_idx = {node: i for i, node in enumerate(nodes)}

    max_weight = max(abs(float(data.get("weight", 1.0))) for _, _, data in G.edges(data=True))
    if max_weight == 0:
        raise ValueError("Al menos una arista debe tener peso distinto de cero.")

    # n qubits y n bits clásicos.
    circuito = Circuit(n, n)

    # Estado inicial |+>^n.
    for i in range(n):
        circuito.H(i)

    # Pytket expresa los ángulos de Rx y Rz en unidades de pi.
    for gamma_layer, beta_layer in zip(gammas, betas):
        for u, v, data in G.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            weight = float(data.get("weight", 1.0)) / max_weight

            circuito.CX(i, j)
            circuito.Rz(gamma_layer * weight / math.pi, j)
            circuito.CX(i, j)

        for i in range(n):
            circuito.Rx(2.0 * beta_layer / math.pi, i)

    # Medición explícita q[i] -> c[i].
    for i in range(n):
        circuito.Measure(i, i)

    return circuito, nodes


def bitstring_from_pytket_key(clave, n_qubits: int) -> str:
    """Normaliza una clave de conteos de Pytket/Nexus (``OutcomeArray``, tupla, str, ...) a bitstring."""
    if hasattr(clave, "to_readouts"):
        readouts = np.asarray(clave.to_readouts(), dtype=int)

        if readouts.ndim != 2 or readouts.shape[0] == 0:
            raise ValueError(f"OutcomeArray inesperado: {clave!r}")

        bits = readouts[0].tolist()

    elif isinstance(clave, np.ndarray):
        bits = np.asarray(clave, dtype=int).reshape(-1).tolist()

    elif isinstance(clave, (tuple, list)):
        bits = [int(bit) for bit in clave]

    elif isinstance(clave, str):
        texto = clave.strip()

        if texto.startswith("0b"):
            return format(int(texto, 2), f"0{n_qubits}b")

        bits_en_texto = [int(bit) for bit in texto if bit in "01"]

        if len(bits_en_texto) != n_qubits:
            raise ValueError(f"No se pudo interpretar la clave {clave!r} como {n_qubits} bits.")

        bits = bits_en_texto

    else:
        raise TypeError(f"Tipo de clave de conteos no soportado: {type(clave).__name__}")

    if len(bits) != n_qubits:
        raise ValueError(f"Se esperaban {n_qubits} bits y se recibieron {len(bits)} en {clave!r}.")

    if any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"La clave contiene valores que no son bits: {clave!r}")

    return "".join(str(bit) for bit in bits)


def counts_from_pytket_results(conteos_crudos, n_qubits: int) -> Counter[str]:
    """Convierte los conteos crudos devueltos por Pytket/Nexus a un ``Counter`` de bitstrings."""
    conteos: Counter[str] = Counter()

    for clave, frecuencia in conteos_crudos.items():
        bitstring = bitstring_from_pytket_key(clave, n_qubits)
        conteos[bitstring] += int(frecuencia)

    return conteos


def results_table(
    G: nx.Graph, counts: Counter[str], nodes: list[str], valor_optimo: float | None = None
) -> pd.DataFrame:
    """Arma una tabla (una fila por bitstring medido) con su probabilidad y valor de corte.

    El valor de corte siempre se calcula con los pesos originales de ``G``
    (vía ``cut_value``), no los normalizados que usa el circuito. Si se pasa
    ``valor_optimo``, agrega la razón de aproximación de cada resultado.
    """
    total = sum(counts.values())
    filas = []

    for bitstring, frecuencia in counts.items():
        valor = cut_value(G, bitstring, nodes)
        fila = {
            "bitstring": bitstring,
            "frecuencia": frecuencia,
            "probabilidad (%)": 100 * frecuencia / total,
            "valor del corte": valor,
        }
        if valor_optimo is not None:
            fila["razón del óptimo (%)"] = 100 * valor / valor_optimo
        filas.append(fila)

    return pd.DataFrame(filas).sort_values("frecuencia", ascending=False).reset_index(drop=True)


def _start_compile_job(
    circuit_ref, backend_config, name: str, project, optimisation_level: int = 2
):
    """Envuelve ``qnx.start_compile_job``, tolerando el parámetro ``programs``/``circuits``
    según la versión de ``qnexus`` instalada."""
    argumentos = {
        "backend_config": backend_config,
        "optimisation_level": optimisation_level,
        "name": name,
        "project": project,
    }

    try:
        return qnx.start_compile_job(programs=[circuit_ref], **argumentos)
    except TypeError as error_programs:
        try:
            return qnx.start_compile_job(circuits=[circuit_ref], **argumentos)
        except TypeError:
            raise RuntimeError(
                "La versión instalada de qnexus no acepta ni 'programs=' ni "
                "'circuits=' en start_compile_job."
            ) from error_programs


def _start_execute_job(circuit_ref, backend_config, name: str, project, n_shots: list[int]):
    """Análogo a ``_start_compile_job``, para el job de ejecución."""
    argumentos = {
        "backend_config": backend_config,
        "n_shots": n_shots,
        "name": name,
        "project": project,
    }

    try:
        return qnx.start_execute_job(programs=[circuit_ref], **argumentos)
    except TypeError as error_programs:
        try:
            return qnx.start_execute_job(circuits=[circuit_ref], **argumentos)
        except TypeError:
            raise RuntimeError(
                "La versión instalada de qnexus no acepta ni 'programs=' ni "
                "'circuits=' en start_execute_job."
            ) from error_programs


def run_qaoa_on_nexus(
    G: nx.Graph,
    gammas: list[float],
    betas: list[float],
    device_name: str = "H2-Emulator",
    project_name: str = "Hackathon-QAOA-H2",
    shots: int = 3000,
    noisy_simulation: bool = False,
    valor_optimo: float | None = None,
    optimisation_level: int = 2,
    timeout: int = 1800,
) -> tuple[Counter[str], list[str], pd.DataFrame]:
    """Compila y corre el circuito QAOA de ``G`` en un dispositivo de Quantinuum vía Nexus.

    Requiere estar autenticado (``qnexus.login()``, innecesario dentro de
    Nexus Lab) y que ``device_name`` esté disponible para la cuenta activa.
    Sube el circuito, lo compila (job de compilación) y lo ejecuta (job de
    ejecución), esperando a que cada uno termine.

    Devuelve ``(conteos, nodes, tabla)``: conteos de bitstrings medidos, el
    orden de nodos, y una tabla ordenada por frecuencia con el valor de corte
    de cada resultado.
    """
    if qnx is None:
        raise ImportError(
            "Este módulo requiere 'qnexus', que no está entre las dependencias del "
            "proyecto (choca con el pandas>=3 que usa quantathon). Instalalo aparte, "
            "en un entorno con acceso a Quantinuum Nexus: pip install qnexus"
        )

    circuito, nodes = build_qaoa_circuit(G, gammas, betas)

    project = qnx.projects.get_or_create(name=project_name)
    qnx.context.set_active_project(project)

    config = qnx.QuantinuumConfig(
        device_name=device_name,
        simulator="state-vector",
        noisy_simulation=noisy_simulation,
    )

    devices_df = qnx.devices.get_all(issuers=["QUANTINUUM"]).df()

    # La columna puede variar entre versiones de qnexus.
    columnas_nombre = [c for c in ("device_name", "name") if c in devices_df.columns]
    if columnas_nombre:
        dispositivos = set(devices_df[columnas_nombre[0]].dropna().astype(str))
        if device_name not in dispositivos:
            raise RuntimeError(
                f"{device_name} no aparece entre los dispositivos disponibles para esta cuenta."
            )

    suffix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + "-" + uuid.uuid4().hex[:8]
    circuit_name = f"qaoa-maxcut-p{len(gammas)}-{suffix}"

    circuit_ref = qnx.circuits.upload(circuit=circuito, name=circuit_name, project=project)

    compile_job_ref = _start_compile_job(
        circuit_ref=circuit_ref,
        backend_config=config,
        name=f"compile-{circuit_name}",
        project=project,
        optimisation_level=optimisation_level,
    )
    qnx.jobs.wait_for(compile_job_ref, timeout=timeout)

    compile_results = qnx.jobs.results(compile_job_ref)
    if len(compile_results) == 0:
        raise RuntimeError("La compilación terminó sin devolver un circuito.")
    compiled_circuit_ref = compile_results[0].get_output()

    execute_job_ref = _start_execute_job(
        circuit_ref=compiled_circuit_ref,
        backend_config=config,
        name=f"execute-{circuit_name}",
        project=project,
        n_shots=[shots],
    )
    qnx.jobs.wait_for(execute_job_ref, timeout=timeout)

    execute_results = qnx.jobs.results(execute_job_ref)
    if len(execute_results) == 0:
        raise RuntimeError("La ejecución terminó sin devolver resultados.")

    conteos_crudos = execute_results[0].get_output().get_counts()
    conteos = counts_from_pytket_results(conteos_crudos, n_qubits=len(nodes))

    if sum(conteos.values()) != shots:
        print(
            f"Advertencia: los conteos suman {sum(conteos.values())} pero se pidieron {shots} shots."
        )

    tabla = results_table(G, conteos, nodes, valor_optimo=valor_optimo)

    return conteos, nodes, tabla
