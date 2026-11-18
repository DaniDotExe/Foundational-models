#!/usr/bin/env python3
"""Comprueba que un .npy es una entrega válida y lo corre en las seis pistas.

    python verificar_entrega.py entregas/Hamilton.npy          prueba corta (1 vuelta o 24 s)
    python verificar_entrega.py entregas/Hamilton.npy --final  la final (5 vueltas o 150 s)
"""
import sys

from src.carrera.exportar import cargar_piloto_npy
from src.carrera.politica import PoliticaNumpy
from src.carrera.evaluacion import (
    evaluar_politica, imprimir_tabla, VUELTAS_PRUEBA, TIEMPO_PRUEBA, VUELTAS_FINAL, TIEMPO_FINAL,
)

if len(sys.argv) not in (2, 3):
    raise SystemExit("Uso: python verificar_entrega.py entregas/Hamilton.npy [--final]")

pesos = cargar_piloto_npy(sys.argv[1])
print("Archivo valido:", sys.argv[1])
final = len(sys.argv) == 3 and sys.argv[2] == "--final"
vueltas, tiempo = (VUELTAS_FINAL, TIEMPO_FINAL) if final else (VUELTAS_PRUEBA, TIEMPO_PRUEBA)
print(f"Prueba: {vueltas} vuelta(s) o {tiempo:.0f} s por pista, conduccion determinista (argmax).")
resultados, _ = evaluar_politica(PoliticaNumpy(pesos), vueltas, tiempo)
imprimir_tabla(resultados)
