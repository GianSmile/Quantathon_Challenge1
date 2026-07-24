"""Constantes de configuración del proyecto."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
LINEAS_TRANSMISION_CSV = DATA_DIR / "LineasDeTransmision.csv"

# Nodos de interés en la región Central Pacífico del SEN (ICE).
REGION_CENTRAL_PACIFICO = [
    "San Miguel",
    "Coronado",
    "El Este",
    "Higuito",
    "Tarbaca",
    "Tejar",
    "Pirris",
    "Parrita",
    "Lindora1",
    "Lindora2",
    "Colima1",
    "Colima2",
]
