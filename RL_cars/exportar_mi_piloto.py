#!/usr/bin/env python3
"""Convierte salidas/mejor_actor.npy en tu entrega: entregas/<NOMBRE_PILOTO>.npy"""
from pathlib import Path

import numpy as np

from src.carrera.exportar import guardar_piloto_npy

NOMBRE_PILOTO = "Hamilton"   # <- cambia esto; el archivo aparecerá como hamilton en el torneo
RUTA = Path("salidas/mejor_actor.npy")

if not RUTA.exists():
    raise SystemExit("Primero ejecuta entrenar.py")
ruta = guardar_piloto_npy(NOMBRE_PILOTO, np.load(RUTA, allow_pickle=False), "entregas")
print("Entrega SOLO este archivo:", ruta)
