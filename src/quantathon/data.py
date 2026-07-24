"""Carga de los datos de líneas de transmisión."""

import pandas as pd

from quantathon.config import LINEAS_TRANSMISION_CSV


def load_lineas_transmision(path=LINEAS_TRANSMISION_CSV) -> pd.DataFrame:
    """Carga el CSV de líneas de transmisión del SEN (ICE)."""
    return pd.read_csv(path)
