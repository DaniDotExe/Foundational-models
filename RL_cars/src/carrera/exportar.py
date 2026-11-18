"""Guardar y cargar pilotos (.npy con los 1703 pesos del actor)."""
from pathlib import Path
import re

import numpy as np

from .politica import validar_pesos


def _nombre_seguro(nombre):
    nombre = re.sub(r"[^A-Za-z0-9_-]+", "_", str(nombre).strip())
    if not nombre:
        raise ValueError("Debes indicar un nombre para tu piloto.")
    return nombre[:60]


def guardar_piloto_npy(nombre, pesos, carpeta="entregas"):
    vector = validar_pesos(pesos).astype(np.float64)
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{_nombre_seguro(nombre)}.npy"
    np.save(ruta, vector)
    print("Piloto guardado en:", ruta)
    return ruta


def cargar_piloto_npy(ruta):
    vector = np.load(ruta, allow_pickle=False)
    return validar_pesos(vector)
