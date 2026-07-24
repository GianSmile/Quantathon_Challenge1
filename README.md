# Quantathon

Modelado y optimización de la red de transmisión eléctrica de Costa Rica
(ICE, ~230/138 kV) como un problema de **Max-Cut**, comparando un enfoque
exacto, dos heurísticas clásicas y un enfoque cuántico (`guppylang`/QAOA)
para el particionamiento en zonas de falla.

Proyecto del equipo **Matrix** para el *Challenge 1: Red Eléctrica
Sostenible, Resiliente y Verde* de Quantathon CR 2026. El reporte completo
(planteamiento, formulación QUBO/Ising, resultados y limitaciones) está en
[`informe.pdf`](informe.pdf).

## Estructura del proyecto

```
.
├── data/                       # Datos crudos (líneas de transmisión)
├── docs/                       # Esquemas de referencia del SEN
├── src/quantathon/
│   ├── config.py                 # Constantes (rutas, nodos de la región)
│   ├── data.py                    # Carga de datos
│   ├── graph.py                   # Construcción del grafo de la red
│   ├── qubo.py                    # Grafo -> matriz de pesos indexada
│   ├── classical.py               # Max-Cut clásico: fuerza bruta (óptimo) y greedy
│   ├── goemans_williamson.py      # Max-Cut vía relajación SDP + redondeo aleatorio
│   ├── qaoa.py                    # Max-Cut vía QAOA (circuito Guppy + ajuste de ángulos)
│   ├── nexus.py                   # QAOA en hardware/emulador Quantinuum vía Nexus (pytket + qnexus)
│   ├── viz.py                     # Visualización del grafo
│   └── __main__.py                # Punto de entrada: compara fuerza bruta, greedy, GW y QAOA
├── scripts/
│   └── nexus_login.py             # Login manual a Nexus + listado de dispositivos disponibles
├── tests/
├── punto_de_entrada.ipynb      # Notebook Deepnote autocontenido (clásicos + QAOA + ZNE + H2-Emulator)
├── informe.pdf                 # Informe del challenge (Matrix, Quantathon CR 2026)
├── graph.png                   # Visualización del grafo de la región Central-Pacífico
└── matriz_pesos.csv / .html    # Matriz de pesos del grafo exportada
```

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- Python 3.14 (uv lo instala automáticamente si no está disponible)

## Instalación

```bash
uv sync
```

También se incluye `requirements.txt` (generado con `uv export`) para quien
prefiera `pip`:

```bash
pip install -r requirements.txt
```

## Uso

```bash
uv run quantathon
# o equivalentemente
uv run python -m quantathon
```

Esto construye el grafo de la región Central Pacífico y resuelve Max-Cut con
los cuatro enfoques, imprimiendo el valor de corte de cada uno y su
porcentaje respecto al óptimo exacto:

- **Fuerza bruta** (`classical.maxcut_brute_force`): prueba todas las
  particiones posibles. Óptimo exacto, exponencial — solo viable para grafos
  chicos como este (12 nodos).
- **Greedy** (`classical.maxcut_greedy`): asigna cada nodo, en orden, al lado
  que maximiza el peso cruzado con los vecinos ya asignados. Polinomial, sin
  garantía de óptimo.
- **Goemans-Williamson** (`goemans_williamson.maxcut_goemans_williamson`):
  relaja el problema a vectores en una esfera (SDP con `cvxpy`) y los
  redondea a una partición binaria con varios hiperplanos aleatorios.
  Garantía teórica de ≥87.8% del óptimo en expectativa.
- **QAOA** (`qaoa.optimize_qaoa` + `qaoa.run_qaoa`): construye el circuito
  cuántico costo-mezclador en Guppy, lo simula en el emulador de `guppylang`,
  y ajusta los ángulos `(gammas, betas)` con `scipy.optimize` (COBYLA) para
  maximizar el corte esperado.

Además guarda la matriz de pesos del grafo como tabla en
`matriz_pesos.csv` y `matriz_pesos.html` (esta última se puede abrir
directamente en el navegador para verla formateada).

## Notebook (`punto_de_entrada.ipynb`)

Versión autocontenida en un único notebook, pensada para correr en
[Deepnote](https://deepnote.com) sin depender del paquete `src/quantathon`.
Reimplementa los mismos algoritmos (fuerza bruta, greedy, Goemans-Williamson
y QAOA) y además agrega lo que no está en el CLI:

- **Zero-Noise Extrapolation (ZNE)** sobre los resultados de QAOA.
- Ejecución opcional en **H2-Emulator** de Quantinuum vía Nexus
  (`!qnx login` + `ejecutar_h2=True`), desactivada por defecto.
- Tablas y gráficas propias para el bloque clásico y el cuántico,
  comparando `p=1` vs `p=2`.

Es el notebook que generó los resultados y las figuras de `informe.pdf`.
Se corre celda a celda; la sección **9. Configuración y ejecución** expone
un diccionario `CONFIG` para elegir qué algoritmos correr y con qué
parámetros (rondas de GW, semillas y `shots` de QAOA, factores de escala de
ZNE, etc.).

## Informe (`informe.pdf`)

Reporte del equipo **Matrix** para el Challenge 1 de Quantathon CR 2026,
con la formulación completa y los resultados obtenidos sobre la instancia
de la región Central-Pacífico (12 nodos, 15 vínculos, óptimo exacto de
296.36 km):

| Método          | Razón de aproximación |
| --------------- | ---------------------- |
| Fuerza bruta     | 100.00% (óptimo)        |
| Greedy           | 98.90%                  |
| Goemans-Williamson (media) | 95.16%        |
| QAOA local, p=1  | 78.57%                  |
| QAOA local, p=2  | 85.07%                  |
| ZNE + corrección, p=2 | 85.56%             |

Conclusión principal: en esta instancia QAOA no supera a los métodos
clásicos (greedy y GW siguen por delante), aunque aumentar la profundidad
`p` sí mejora consistentemente la razón de aproximación. El informe detalla
además las limitaciones (sin réplicas ni desviación estándar, sin
restricciones de conectividad/N-1, ejecución solo en emulador) y el trabajo
pendiente. También incluye la relación del proyecto con los ODS 7, 9 y 13.

## Ejecutar QAOA en Quantinuum (Nexus)

`nexus.py` construye el mismo circuito QAOA pero como `pytket.Circuit`, y lo
sube/compila/ejecuta en un dispositivo real o emulador de Quantinuum a
través de [Nexus](https://nexus.quantinuum.com/). **No** se instala como
dependencia del proyecto: el paquete `qnexus` fija `pandas<3`, y choca con el
`pandas>=3.0.3` que usa el resto de quantathon (`uv add qnexus` falla por
esto). Está pensado para correr en un entorno aparte con acceso a Nexus —
típicamente el propio "Nexus Lab" de Quantinuum, que ya trae `qnexus` y
`pytket` preinstalados—, no vía `uv run quantathon`.

### Entorno aparte

Fuera de Nexus Lab hay que crear un entorno separado del `.venv` principal
(no se puede instalar `qnexus` ahí por el choque de `pandas` explicado
arriba):

```bash
python3.14 -m venv .venv-nexus
source .venv-nexus/bin/activate
pip install -e . --no-deps        # trae quantathon.nexus / .classical sin arrastrar pandas>=3
pip install "pandas<3" networkx numpy pytket qnexus
```

### Login

**Corré el login en tu propia terminal, no a través de un agente/chat** —
así el token de autenticación nunca queda expuesto en una transcripción.
`scripts/nexus_login.py` hace justamente eso: dispara el login (abre
navegador o muestra un link/código) y después lista los dispositivos
disponibles para tu cuenta:

```bash
source .venv-nexus/bin/activate
python scripts/nexus_login.py
```

El token queda guardado en `~/.qnx/auth/`, así que no hace falta repetir el
login en cada proceso nuevo mientras uses la misma máquina.

### Correr el circuito

```python
from quantathon.nexus import run_qaoa_on_nexus

conteos, nodes, tabla = run_qaoa_on_nexus(
    G,
    gammas=GAMMAS_OPT,
    betas=BETAS_OPT,
    device_name="H2-Emulator",   # o el dispositivo real disponible en tu cuenta
    shots=3000,
    valor_optimo=mejor_valor,    # opcional, para reportar % del óptimo
)
```

`run_qaoa_on_nexus` sube el circuito, lanza el job de compilación y el de
ejecución (esperando a que cada uno termine), y devuelve los conteos de
bitstrings ya normalizados junto con una tabla ordenada por frecuencia con
el valor de corte de cada resultado — evaluado siempre con los pesos
originales del grafo, no los normalizados que usa el circuito.

## Tests

```bash
uv run pytest
```
