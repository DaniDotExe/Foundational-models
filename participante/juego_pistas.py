"""Simulador UIS Grand Prix Foundationals Models (Pygame + Física RL_cars).

Reutiliza el motor físico, geometría, colisiones y circuitos de `RL_cars`,
adaptados a:
- 5 sensores de rayos normalizados: [izq, diag_izq, frente, diag_der, der]
- 4 acciones discretas: 0=recto, 1=izquierda, 2=derecha, 3=frenar
- 3 Pistas de Entrenamiento (A1, A2, A3) y 3 Pistas de Evaluación Ocultas (B1, B2, B3)
- 4 Modos de ejecución:
  [1] Grabar Jugadas (Pistas A1, A2, A3) -> data/mis_jugadas.json
  [2] Entrenar Piloto (Solo si ESTRATEGIA == "LORA") -> POST /entrenar_lora
  [3] Modo Autónomo - Prueba (Pistas A1, A2, A3) -> POST /decidir_accion
  [4] Evaluación Oficial (Pistas B1, B2, B3) -> POST /registrar_puntaje
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from bridge import ServerBridge
import config_estudiante as cfg

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
JUGADAS_PATH = DATA_DIR / "mis_jugadas.json"
RECORD_EVERY_N_FRAMES = 1
DECIDE_EVERY_N_FRAMES = 1

# ---------------------------------------------------------------------------
# Constantes físicas extraídas de RL_cars/src/carrera/simulador.py
# ---------------------------------------------------------------------------
DT = 0.06
VELOCIDAD_MAX = 7.6
ACELERACION = 5.2
FRENO = 8.8
ARRASTRE = 0.20
GIRO_MAX = 2.20
RADIO_CHECKPOINT = 1.45
RADIO_CARRO = 0.28
MARGEN_CONTACTO = 0.05
PENALIZACION_FUERA = 1.0
PENALIZACION_CHOQUE = 2.0

ALCANCE_RAYO = 8.0
PASO_RAYO = 0.20
ANGULOS_RAYOS = (-math.pi / 2.0, -math.pi / 4.0, 0.0, math.pi / 4.0, math.pi / 2.0)
NOMBRES_RAYOS = ("izq", "diag_izq", "frente", "diag_der", "der")

TIEMPO_LIMITE_PRUEBA = 28.0
VUELTAS_OBJETIVO = 1

# 4 acciones discretas del MVP (0..3)
ACCIONES_DISCRETAS: List[Tuple[str, float, float, float]] = [
    ("Recto + Acelerar", 1.0, 0.0, 0.0),      # 0
    ("Izquierda + Acelerar", 1.0, -1.0, 0.0), # 1
    ("Derecha + Acelerar", 1.0, 1.0, 0.0),    # 2
    ("Frenar", 0.0, 0.0, 1.0),                # 3
]


# ---------------------------------------------------------------------------
# Geometría de pistas (RL_cars/src/carrera/geometria.py)
# ---------------------------------------------------------------------------
def limitar(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x


def envolver_angulo(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def proyeccion_en_segmento(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float]:
    x, y = bx - ax, by - ay
    den = x * x + y * y
    if den <= 1e-12:
        return ax, ay, 0.0
    t = limitar(((px - ax) * x + (py - ay) * y) / den, 0.0, 1.0)
    return ax + t * x, ay + t * y, t


def punto_mas_cercano_pista(
    px: float, py: float, puntos: List[Tuple[float, float]]
) -> Tuple[float, float, float]:
    """Devuelve (qx, qy, distancia) del punto más cercano sobre la línea central."""
    mejor_d = float("inf")
    mejor_qx, mejor_qy = px, py
    n = len(puntos)
    for i in range(n):
        ax, ay = puntos[i]
        bx, by = puntos[(i + 1) % n]
        qx, qy, _ = proyeccion_en_segmento(px, py, ax, ay, bx, by)
        d = math.hypot(px - qx, py - qy)
        if d < mejor_d:
            mejor_d = d
            mejor_qx, mejor_qy = qx, qy
    return mejor_qx, mejor_qy, mejor_d


def error_lateral_firmado(px: float, py: float, puntos: List[Tuple[float, float]]) -> float:
    mejor = float("inf")
    firmado = 0.0
    n = len(puntos)
    for i in range(n):
        ax, ay = puntos[i]
        bx, by = puntos[(i + 1) % n]
        qx, qy, _ = proyeccion_en_segmento(px, py, ax, ay, bx, by)
        d = math.hypot(px - qx, py - qy)
        if d < mejor:
            cruz = (bx - ax) * (py - qy) - (by - ay) * (px - qx)
            mejor = d
            firmado = (1.0 if cruz >= 0 else -1.0) * d
    return firmado


def longitud_pista(puntos: List[Tuple[float, float]]) -> float:
    n = len(puntos)
    return sum(
        math.hypot(
            puntos[(i + 1) % n][0] - puntos[i][0],
            puntos[(i + 1) % n][1] - puntos[i][1],
        )
        for i in range(n)
    )


def progreso_en_pista(px: float, py: float, puntos: List[Tuple[float, float]]) -> float:
    n = len(puntos)
    mejor = float("inf")
    progreso = 0.0
    acumulado = 0.0
    for i in range(n):
        ax, ay = puntos[i]
        bx, by = puntos[(i + 1) % n]
        qx, qy, t = proyeccion_en_segmento(px, py, ax, ay, bx, by)
        d = math.hypot(px - qx, py - qy)
        largo = math.hypot(bx - ax, by - ay)
        if d < mejor:
            mejor = d
            progreso = acumulado + t * largo
        acumulado += largo
    return progreso


def avance(progreso_anterior: float, progreso_nuevo: float, longitud: float) -> float:
    delta = progreso_nuevo - progreso_anterior
    if delta > longitud / 2.0:
        delta -= longitud
    elif delta < -longitud / 2.0:
        delta += longitud
    return delta


# ---------------------------------------------------------------------------
# 6 Circuitos oficiales (3 Pistas A de entrenamiento + 3 Pistas B ocultas)
# ---------------------------------------------------------------------------
PISTAS: Dict[str, Dict[str, Any]] = {
    "P1": {
        "nombre": "P1 - Óvalo (Entrenamiento)",
        "tipo": "TRAIN",
        "dificultad": 1,
        "ancho_medio": 2.35,
        "puntos": [
            (-8.0, 0.0), (-6.5, -4.0), (-2.5, -5.6), (2.5, -5.6), (6.5, -4.0),
            (8.0, 0.0), (6.5, 4.0), (2.5, 5.6), (-2.5, 5.6), (-6.5, 4.0),
        ],
        "obstaculos": [],
        "obstaculos_moviles": [],
    },
    "P2": {
        "nombre": "P2 - Curvas Amplias (Entrenamiento)",
        "tipo": "TRAIN",
        "dificultad": 2,
        "ancho_medio": 2.25,
        "puntos": [
            (-8.3, -0.6), (-6.3, -4.4), (-1.5, -5.5), (3.6, -4.8), (7.5, -2.0),
            (8.1, 2.1), (5.0, 5.3), (0.2, 5.6), (-3.9, 4.4), (-7.5, 2.3),
        ],
        "obstaculos": [],
        "obstaculos_moviles": [],
    },
    "P3": {
        "nombre": "P3 - Chicane Técnica (Entrenamiento)",
        "tipo": "TRAIN",
        "dificultad": 3,
        "ancho_medio": 2.10,
        "puntos": [
            (-8.2, 0.0), (-6.4, -4.4), (-2.0, -5.5), (0.4, -2.0), (3.0, -5.1), (7.0, -3.5),
            (8.2, 0.5), (5.8, 4.8), (1.8, 5.5), (-0.7, 2.0), (-3.7, 5.0), (-7.1, 3.2),
        ],
        "obstaculos": [],
        "obstaculos_moviles": [],
    },
    "B1": {
        "nombre": "B1 - La Serpiente (Evaluación)",
        "tipo": "TEST",
        "dificultad": 4,
        "ancho_medio": 1.95,
        "puntos": [
            (-8.4, 1.2), (-7.0, -4.6), (-3.1, -5.5), (-0.3, -2.2), (2.4, -5.3), (6.6, -4.0),
            (8.3, -0.8), (6.0, 1.3), (8.0, 4.3), (3.7, 5.7), (0.7, 2.5), (-2.2, 5.4), (-6.5, 4.4),
        ],
        "obstaculos": [],
        "obstaculos_moviles": [],
    },
    "B2": {
        "nombre": "B2 - La Cantera con Obstáculos (Evaluación)",
        "tipo": "TEST",
        "dificultad": 5,
        "ancho_medio": 1.90,
        "puntos": [
            (-8.5, 0.8), (-7.1, -4.5), (-3.0, -5.6), (1.3, -5.2), (4.7, -3.7), (7.9, -1.3),
            (7.3, 2.0), (4.6, 4.9), (0.8, 5.6), (-1.2, 2.5), (-4.4, 5.0), (-7.8, 3.5),
        ],
        "obstaculos": [
            (-4.4, -4.55, 0.62), (2.8, -4.35, 0.68), (6.75, 0.55, 0.72),
            (2.85, 4.85, 0.62), (-2.55, 3.70, 0.65), (-7.00, 2.20, 0.58),
        ],
        "obstaculos_moviles": [],
    },
    "B3": {
        "nombre": "B3 - Hora Pico Móvil (Evaluación)",
        "tipo": "TEST",
        "dificultad": 6,
        "ancho_medio": 2.10,
        "puntos": [
            (-8.4, 0.0), (-7.3, -4.0), (-3.8, -5.6), (0.0, -5.6), (3.8, -5.6), (7.3, -4.0),
            (8.4, 0.0), (7.3, 4.0), (3.8, 5.6), (0.0, 4.9), (-3.8, 5.6), (-7.3, 4.0),
        ],
        "obstaculos": [],
        "obstaculos_moviles": [
            (-1.90, -7.70, -1.90, -3.50, 0.65, 4.0, 0.00),
            (1.90, -3.50, 1.90, -7.70, 0.60, 5.0, 0.00),
            (9.87, 2.56, 5.83, 1.44, 0.70, 4.5, 0.25),
            (2.28, 3.18, 1.52, 7.32, 0.65, 5.5, 0.60),
            (-5.83, 1.44, -9.87, 2.56, 0.60, 3.6, 0.40),
        ],
    },
}

PISTAS_TRAIN = ["P1", "P2", "P3"]
PISTAS_TEST = ["B1", "B2", "B3"]

for _p in PISTAS.values():
    _p["longitud"] = longitud_pista(_p["puntos"])


def obstaculos_en(pista: Dict[str, Any], tiempo: float) -> List[Tuple[float, float, float]]:
    obstaculos = list(pista["obstaculos"])
    for ax, ay, bx, by, radio, periodo, fase in pista["obstaculos_moviles"]:
        u = 0.5 - 0.5 * math.cos(2.0 * math.pi * (tiempo / periodo + fase))
        obstaculos.append((ax + u * (bx - ax), ay + u * (by - ay), radio))
    return obstaculos


# ---------------------------------------------------------------------------
# Estado físico del carro y sensores de 5 rayos
# ---------------------------------------------------------------------------
@dataclass
class EstadoCarro:
    x: float
    y: float
    angulo: float
    velocidad: float = 0.0
    siguiente_checkpoint: int = 1
    checkpoints: int = 0
    vueltas: int = 0
    tiempo: float = 0.0
    tiempo_fuera: float = 0.0
    colisiones_obstaculos: int = 0
    pasos: int = 0
    progreso: float = 0.0
    progreso_acumulado: float = 0.0
    max_progreso: float = 0.0
    tiempo_penalizacion: float = 0.0
    contactos: Tuple[int, ...] = ()

    @property
    def tiempo_carrera(self) -> float:
        return self.tiempo + self.tiempo_penalizacion


def estado_inicial(clave_pista: str) -> EstadoCarro:
    puntos = PISTAS[clave_pista]["puntos"]
    x0, y0 = puntos[0]
    x1, y1 = puntos[1]
    e = EstadoCarro(x=x0, y=y0, angulo=math.atan2(y1 - y0, x1 - x0))
    e.siguiente_checkpoint = 1
    e.progreso = progreso_en_pista(x0, y0, puntos)
    return e


def _resolver_obstaculos(
    estado: EstadoCarro, obstaculos: List[Tuple[float, float, float]]
) -> int:
    choques = 0
    contactos = []
    for i, (ox, oy, r) in enumerate(obstaculos):
        dx, dy = estado.x - ox, estado.y - oy
        d = math.hypot(dx, dy)
        limite = r + RADIO_CARRO
        if d < limite:
            if d < 1e-9:
                dx, dy, d = 1.0, 0.0, 1.0
            estado.x = ox + dx / d * limite
            estado.y = oy + dy / d * limite
            estado.velocidad *= 0.24
            if i not in estado.contactos:
                choques += 1
            contactos.append(i)
        elif d < limite + MARGEN_CONTACTO and i in estado.contactos:
            contactos.append(i)
    estado.contactos = tuple(contactos)
    estado.colisiones_obstaculos += choques
    return choques


def paso_simulacion(estado: EstadoCarro, accion: int, clave_pista: str) -> Dict[str, Any]:
    pista = PISTAS[clave_pista]
    puntos = pista["puntos"]
    nombre, acel, giro, freno = ACCIONES_DISCRETAS[int(accion) % len(ACCIONES_DISCRETAS)]
    arrastre_ef = ARRASTRE
    e = estado

    e.velocidad = limitar(
        e.velocidad + (acel * ACELERACION - freno * FRENO - arrastre_ef * e.velocidad) * DT,
        0.0,
        VELOCIDAD_MAX,
    )
    e.angulo = envolver_angulo(
        e.angulo + giro * GIRO_MAX * (0.28 + 0.72 * e.velocidad / VELOCIDAD_MAX) * DT
    )
    e.x += math.cos(e.angulo) * e.velocidad * DT
    e.y += math.sin(e.angulo) * e.velocidad * DT

    # Paredes sólidas en los bordes del mapa (sin voltear la dirección)
    limite_x, limite_y = 11.0, 9.9
    if abs(e.x) >= limite_x or abs(e.y) >= limite_y:
        e.x = limitar(e.x, -limite_x, limite_x)
        e.y = limitar(e.y, -limite_y, limite_y)
        e.velocidad *= 0.65

    e.tiempo += DT
    e.pasos += 1

    error_lat = abs(error_lateral_firmado(e.x, e.y, puntos))
    fuera = error_lat > pista["ancho_medio"]
    penalizacion = 0.0
    if fuera:
        e.tiempo_fuera += DT
        e.velocidad *= 0.76
        penalizacion += PENALIZACION_FUERA * DT

    choques = _resolver_obstaculos(e, obstaculos_en(pista, e.tiempo))
    penalizacion += PENALIZACION_CHOQUE * choques
    e.tiempo_penalizacion += penalizacion

    progreso_nuevo = progreso_en_pista(e.x, e.y, puntos)
    _ = avance(e.progreso, progreso_nuevo, pista["longitud"])
    e.progreso = progreso_nuevo

    tx, ty = puntos[e.siguiente_checkpoint]
    radio_cp = max(RADIO_CHECKPOINT, pista["ancho_medio"] + RADIO_CARRO)
    if math.hypot(e.x - tx, e.y - ty) <= radio_cp:
        e.siguiente_checkpoint = (e.siguiente_checkpoint + 1) % len(puntos)
        e.checkpoints += 1
        if e.siguiente_checkpoint == 1:
            e.vueltas += 1

    # Progreso porcentual monotónico basado en checkpoints reales de la vuelta
    n_pts = len(puntos)
    if e.vueltas >= VUELTAS_OBJETIVO:
        e.max_progreso = 100.0
    else:
        prev_cp = (e.siguiente_checkpoint - 1) % n_pts
        ax, ay = puntos[prev_cp]
        bx, by = puntos[e.siguiente_checkpoint]
        _, _, t_seg = proyeccion_en_segmento(e.x, e.y, ax, ay, bx, by)
        pct_actual = ((e.checkpoints + t_seg) / float(n_pts)) * 100.0
        if pct_actual > e.max_progreso:
            e.max_progreso = pct_actual

    return {
        "nombre_accion": nombre,
        "fuera_pista": fuera,
        "colision": choques > 0,
    }


def calcular_sensores_rayos(
    estado: EstadoCarro, clave_pista: str
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """Lanza 5 rayos normalizados [izq, diag_izq, frente, diag_der, der] hasta ALCANCE_RAYO."""
    pista = PISTAS[clave_pista]
    puntos = pista["puntos"]
    ancho = pista["ancho_medio"]
    obstaculos = obstaculos_en(pista, estado.tiempo)
    carro_dentro = abs(error_lateral_firmado(estado.x, estado.y, puntos)) <= ancho

    sensores: List[float] = []
    impactos: List[Tuple[float, float]] = []

    for offset in ANGULOS_RAYOS:
        ang = estado.angulo + offset
        dx = math.cos(ang)
        dy = math.sin(ang)
        hit_dist = ALCANCE_RAYO
        hit_x = estado.x + dx * ALCANCE_RAYO
        hit_y = estado.y + dy * ALCANCE_RAYO

        dist = PASO_RAYO
        while dist <= ALCANCE_RAYO + 1e-9:
            rx = estado.x + dx * dist
            ry = estado.y + dy * dist
            toco_obstaculo = False
            for ox, oy, r in obstaculos:
                if math.hypot(rx - ox, ry - oy) <= r + RADIO_CARRO * 0.5:
                    toco_obstaculo = True
                    break
            dentro_ahora = abs(error_lateral_firmado(rx, ry, puntos)) <= ancho

            # Dentro de pista: detecta borde exterior. Fuera de pista: detecta borde al volver a entrar u obstáculos.
            toco_borde = (carro_dentro and not dentro_ahora) or (not carro_dentro and dentro_ahora)
            if toco_obstaculo or toco_borde:
                hit_dist = max(0.0, dist - PASO_RAYO * 0.5)
                hit_x = estado.x + dx * hit_dist
                hit_y = estado.y + dy * hit_dist
                break
            dist += PASO_RAYO

        sensores.append(round(limitar(hit_dist / ALCANCE_RAYO, 0.0, 1.0), 4))
        impactos.append((hit_x, hit_y))

    return sensores, impactos


def calcular_progreso_pct(estado: EstadoCarro, clave_pista: str) -> float:
    if estado.vueltas >= VUELTAS_OBJETIVO:
        return 100.0
    _ = clave_pista
    return round(limitar(estado.max_progreso, 0.0, 99.9), 2)


def piloto_guia_local(sensores: List[float], velocidad: float) -> int:
    """Heurística de demostración (0=recto, 1=izq, 2=der, 3=frenar)."""
    izq, diag_izq, frente, diag_der, der = sensores
    if frente < 0.22 and velocidad > 5.0:
        return 3
    if frente < 0.35:
        return 1 if (diag_izq + 0.30 * izq) >= (diag_der + 0.30 * der) else 2
    diff = (diag_izq + 0.25 * izq) - (diag_der + 0.25 * der)
    if diff > 0.07:
        return 1
    if diff < -0.07:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Persistencia de jugadas locales
# ---------------------------------------------------------------------------
def cargar_jugadas() -> List[Dict[str, Any]]:
    if not JUGADAS_PATH.exists():
        return []
    try:
        data = json.loads(JUGADAS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def guardar_jugadas(jugadas: List[Dict[str, Any]]) -> None:
    JUGADAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    JUGADAS_PATH.write_text(
        json.dumps(jugadas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def velocidad_normalizada(velocidad: float) -> float:
    return round(limitar(velocidad / VELOCIDAD_MAX, 0.0, 1.0), 4)


def info_checkpoint(estado: EstadoCarro, clave_pista: str) -> Tuple[float, float]:
    """Devuelve (distancia_norm [0..1], angulo_relativo_norm [-1..1]) al próximo checkpoint."""
    puntos = PISTAS[clave_pista]["puntos"]
    tx, ty = puntos[estado.siguiente_checkpoint % len(puntos)]
    dx, dy = tx - estado.x, ty - estado.y
    dist_norm = round(limitar(math.hypot(dx, dy) / 12.0, 0.0, 1.0), 4)
    ang_rel = envolver_angulo(math.atan2(dy, dx) - estado.angulo) / math.pi
    return dist_norm, round(limitar(ang_rel, -1.0, 1.0), 4)


def conteo_por_pista(jugadas: List[Dict[str, Any]]) -> Dict[str, int]:
    conteo = {k: 0 for k in PISTAS_TRAIN}
    for j in jugadas:
        p = str(j.get("pista", ""))
        if p in conteo:
            conteo[p] += 1
        else:
            conteo[p] = conteo.get(p, 0) + 1
    return conteo


def generar_jugadas_semilla(max_por_pista: int = 40) -> List[Dict[str, Any]]:
    """Genera jugadas limpias de demostración (~120 muestras) en A1, A2, A3."""
    muestras: List[Dict[str, Any]] = []
    ventana = max(1, min(8, int(cfg.CONFIG_DATOS.get("ventana_historia", 5))))
    for clave in PISTAS_TRAIN:
        est = estado_inicial(clave)
        hist: List[int] = [0] * ventana
        frame = 0
        capturadas = 0
        while est.tiempo_carrera < TIEMPO_LIMITE_PRUEBA and est.vueltas < 3 and capturadas < max_por_pista:
            sensores, _ = calcular_sensores_rayos(est, clave)
            accion = piloto_guia_local(sensores, est.velocidad)
            if frame % RECORD_EVERY_N_FRAMES == 0:
                cp_dist, cp_ang = info_checkpoint(est, clave)
                muestras.append({
                    "pista": clave,
                    "sensores": sensores,
                    "velocidad": velocidad_normalizada(est.velocidad),
                    "cp_dist": cp_dist,
                    "cp_ang": cp_ang,
                    "historial": hist[-ventana:],
                    "accion": accion,
                    "estado": {
                        "x": round(est.x, 3),
                        "y": round(est.y, 3),
                        "angulo": round(est.angulo, 4),
                        "velocidad": round(est.velocidad, 2),
                        "tiempo": round(est.tiempo, 2),
                        "siguiente_checkpoint": int(est.siguiente_checkpoint),
                        "checkpoints": int(est.checkpoints),
                        "vueltas": int(est.vueltas),
                    },
                })
                capturadas += 1
            paso_simulacion(est, accion, clave)
            hist.append(accion)
            frame += 1
    return muestras


# ---------------------------------------------------------------------------
# Ejecución Headless / CLI para modos 2, 3 y 4
# ---------------------------------------------------------------------------
def ejecutar_entrenamiento_lora(bridge: ServerBridge) -> Dict[str, Any]:
    if cfg.ESTRATEGIA.upper().strip() != "LORA":
        raise RuntimeError(
            f"ESTRATEGIA actual en config_estudiante.py es '{cfg.ESTRATEGIA}'. "
            "Cámbiala a 'LORA' para usar el modo [2] Entrenar Piloto."
        )
    jugadas = cargar_jugadas()
    if not jugadas:
        raise RuntimeError("No hay jugadas en data/mis_jugadas.json. Ejecuta primero el modo [1].")
    print(f"[Modo 2] Enviando {len(jugadas)} jugadas disponibles para entrenar LoRA ({bridge.student_id})...")
    res = bridge.entrenar_lora(jugadas, cfg.CONFIG_LORA, cfg.CONFIG_DATOS)
    print(
        f"[Modo 2 OK] Tiempo: {res['tiempo_entrenamiento_s']}s | "
        f"Ejemplos: {res['ejemplos_entrenados']} | Loss final: {res['loss_final']}"
    )
    return res


def simular_pista_autonoma_headless(
    bridge: ServerBridge,
    clave_pista: str,
    jugadas_ref: List[Dict[str, Any]],
) -> Dict[str, Any]:
    est = estado_inicial(clave_pista)
    ventana = max(1, min(8, int(cfg.CONFIG_DATOS.get("ventana_historia", 5))))
    hist: List[int] = [0] * ventana
    accion_actual = 0
    frame = 0
    latencias_rtt_ms: List[float] = []
    latencias_gpu_ms: List[float] = []
    trayectoria: List[List[float]] = []

    while est.tiempo_carrera < TIEMPO_LIMITE_PRUEBA and est.vueltas < VUELTAS_OBJETIVO:
        if frame % DECIDE_EVERY_N_FRAMES == 0:
            sensores, _ = calcular_sensores_rayos(est, clave_pista)
            cp_dist, cp_ang = info_checkpoint(est, clave_pista)
            resp = bridge.decidir_accion(
                estrategia=cfg.ESTRATEGIA,
                estado_actual=sensores,
                historial=hist[-ventana:],
                jugadas_referencia=jugadas_ref,
                config_icl=cfg.CONFIG_ICL,
                config_datos=cfg.CONFIG_DATOS,
                velocidad=velocidad_normalizada(est.velocidad),
                cp_dist=cp_dist,
                cp_ang=cp_ang,
            )
            accion_actual = int(resp["accion"])
            latencias_rtt_ms.append(float(resp["rtt_ms"]))
            latencias_gpu_ms.append(float(resp["latencia_gpu_ms"]))

        trayectoria.append([round(est.x, 2), round(est.y, 2), round(est.angulo, 3), int(accion_actual)])
        paso_simulacion(est, accion_actual, clave_pista)
        hist.append(accion_actual)
        frame += 1

    lat_acum_s = round(sum(latencias_rtt_ms) / 1000.0, 4)
    lat_prom_ms = round(sum(latencias_rtt_ms) / max(1, len(latencias_rtt_ms)), 2)
    return {
        "pista": clave_pista,
        "progreso_pct": calcular_progreso_pct(est, clave_pista),
        "tiempo_fisico_s": round(est.tiempo_carrera, 3),
        "latencia_acumulada_s": lat_acum_s,
        "latencia_promedio_ms": lat_prom_ms,
        "latencia_gpu_prom_ms": round(sum(latencias_gpu_ms) / max(1, len(latencias_gpu_ms)), 2),
        "completada": est.vueltas >= VUELTAS_OBJETIVO,
        "choques": est.colisiones_obstaculos,
        "tiempo_fuera_s": round(est.tiempo_fuera, 3),
        "trayectoria": trayectoria,
    }


def ejecutar_evaluacion_oficial_headless(bridge: ServerBridge) -> Dict[str, Any]:
    jugadas_ref = cargar_jugadas()
    if cfg.ESTRATEGIA.upper().strip() == "ICL" and not jugadas_ref:
        raise RuntimeError("No hay jugadas en data/mis_jugadas.json para ICL.")

    resultados: Dict[str, Dict[str, Any]] = {}
    for clave in PISTAS_TEST:
        print(f"[Modo 4] Evaluando {PISTAS[clave]['nombre']} ({cfg.ESTRATEGIA})...")
        r = simular_pista_autonoma_headless(bridge, clave, jugadas_ref)
        resultados[clave] = r
        print(
            f"  -> {clave}: progreso={r['progreso_pct']:.1f}% | "
            f"tiempo_fisico={r['tiempo_fisico_s']:.2f}s | "
            f"latencia={r['latencia_acumulada_s']:.3f}s ({r['latencia_promedio_ms']:.1f}ms)"
        )

    num_ej = (
        len(jugadas_ref)
        if cfg.ESTRATEGIA.upper().strip() == "ICL"
        else min(len(jugadas_ref), int(cfg.CONFIG_LORA.get("num_ejemplos", 200)))
    )
    respuesta = bridge.registrar_puntaje(cfg.ESTRATEGIA, resultados, num_ejemplos=num_ej)
    reg = respuesta["registro"]
    pts_lbl = reg.get("puntos_f1", reg.get("score_final", 0.0))
    print(
        f"\n[RANKING] Posición #{respuesta['posicion']}/{respuesta['total_estudiantes']} | "
        f"Puntos F1: {pts_lbl} | % Promedio B: {reg['progreso_promedio_pct']}%"
    )
    return respuesta


# ---------------------------------------------------------------------------
# Interfaz Visual Pygame (Menú + 4 Modos)
# ---------------------------------------------------------------------------
class JuegoPygame:
    ANCHO = 1220
    ALTO = 780
    AREA_PISTA_W = 860

    BG = (8, 12, 22)
    PANEL_BG = (15, 23, 38)
    BORDE = (41, 53, 74)
    ASFALTO = (28, 36, 52)
    BORDE_PISTA = (68, 82, 108)
    TEXTO = (241, 245, 249)
    MUTED = (148, 163, 184)
    ACCENT = (20, 184, 166)
    CAR_COLOR = (53, 250, 224)
    WARN = (245, 158, 11)
    BAD = (244, 63, 94)
    OK = (74, 222, 128)

    def __init__(self, bridge: ServerBridge) -> None:
        import pygame

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption("UIS Grand Prix Foundationals Models — Simulador")
        self.screen = pygame.display.set_mode((self.ANCHO, self.ALTO))
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("consolas,segoeui,arial", 24, bold=True)
        self.font_h2 = pygame.font.SysFont("consolas,segoeui,arial", 18, bold=True)
        self.font_body = pygame.font.SysFont("consolas,segoeui,arial", 15)
        self.font_small = pygame.font.SysFont("consolas,segoeui,arial", 13)

        self.bridge = bridge
        self.estado_menu_msg = "Listo. Selecciona un modo (1-4)."
        self.salud_servidor: Optional[Dict[str, Any]] = None
        self._bg_lock = threading.Lock()
        self._bg_queue: List[str] = []
        self._bg_active_track: Optional[str] = None
        self._bg_active_pct: float = 0.0
        self._bg_active_frames: int = 0
        self._bg_active_pista_pct: float = 0.0
        self._bg_active_rtt: float = 0.0
        self._bg_active_estrategia: str = ""
        self._bg_cancel_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._bg_results: Dict[str, Dict[str, Any]] = {}
        self.loras_disponibles: List[str] = self.bridge.listar_adaptadores()
        self.lora_activo: Optional[str] = self.loras_disponibles[-1] if self.loras_disponibles else None
        self._train_lock = threading.Lock()
        self._train_cancel_ev = threading.Event()
        self._train_state: Dict[str, Any] = {
            "running": False,
            "name": "",
            "pct": 0.0,
            "epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "t_start": 0.0,
        }
        self._chequear_salud_silenciosa()

    def _chequear_salud_silenciosa(self) -> None:
        try:
            self.salud_servidor = self.bridge.verificar_salud()
        except Exception:
            self.salud_servidor = None

    def _w2s(self, x: float, y: float) -> Tuple[int, int, float]:
        escala = min(self.AREA_PISTA_W / 23.0, self.ALTO / 16.5)
        sx = int(self.AREA_PISTA_W / 2.0 + x * escala)
        sy = int(self.ALTO / 2.0 + y * escala)
        return sx, sy, escala

    def _dibujar_pista_y_carro(
        self,
        clave_pista: str,
        estado: EstadoCarro,
        sensores: List[float],
        impactos: List[Tuple[float, float]],
    ) -> None:
        pg = self.pygame
        pista = PISTAS[clave_pista]
        _, _, escala = self._w2s(0.0, 0.0)

        pg.draw.rect(self.screen, self.BG, (0, 0, self.AREA_PISTA_W, self.ALTO))

        pts_screen = [(self._w2s(px, py)[0], self._w2s(px, py)[1]) for px, py in pista["puntos"]]
        ancho_px = max(12, int(pista["ancho_medio"] * 2.0 * escala))
        pg.draw.lines(self.screen, self.BORDE_PISTA, True, pts_screen, ancho_px + 12)
        for sx, sy in pts_screen:
            pg.draw.circle(self.screen, self.BORDE_PISTA, (sx, sy), (ancho_px + 12) // 2)
        pg.draw.lines(self.screen, self.ASFALTO, True, pts_screen, ancho_px)
        for sx, sy in pts_screen:
            pg.draw.circle(self.screen, self.ASFALTO, (sx, sy), ancho_px // 2)

        # Línea central sutil y checkpoints
        pg.draw.lines(self.screen, (45, 58, 82), True, pts_screen, 2)
        for idx, (sx, sy) in enumerate(pts_screen):
            if idx == 0:
                pg.draw.circle(self.screen, self.OK, (sx, sy), 8)
            elif idx == estado.siguiente_checkpoint:
                pg.draw.circle(self.screen, self.WARN, (sx, sy), 7, 2)
            else:
                pg.draw.circle(self.screen, (71, 85, 105), (sx, sy), 3)

        # Guías obstáculos móviles
        for ax, ay, bx, by, _, _, _ in pista["obstaculos_moviles"]:
            pa = self._w2s(ax, ay)[:2]
            pb = self._w2s(bx, by)[:2]
            pg.draw.line(self.screen, (140, 70, 60), pa, pb, 2)

        # Obstáculos fijos y móviles
        n_fijos = len(pista["obstaculos"])
        for i, (ox, oy, r) in enumerate(obstaculos_en(pista, estado.tiempo)):
            cx, cy, _ = self._w2s(ox, oy)
            color = self.WARN if i < n_fijos else (255, 130, 105)
            pg.draw.circle(self.screen, color, (cx, cy), max(4, int(r * escala)))

        # Rayos de los 5 sensores + rayo al próximo checkpoint
        car_x, car_y, _ = self._w2s(estado.x, estado.y)
        cp_pt = pista["puntos"][estado.siguiente_checkpoint % len(pista["puntos"])]
        cp_sx, cp_sy, _ = self._w2s(cp_pt[0], cp_pt[1])
        pg.draw.line(self.screen, (56, 189, 248), (car_x, car_y), (cp_sx, cp_sy), 2)
        pg.draw.circle(self.screen, (56, 189, 248), (cp_sx, cp_sy), 5)

        for val, (hx, hy) in zip(sensores, impactos):
            ix, iy, _ = self._w2s(hx, hy)
            color_rayo = self.OK if val > 0.5 else (self.WARN if val > 0.25 else self.BAD)
            pg.draw.line(self.screen, color_rayo, (car_x, car_y), (ix, iy), 2)
            pg.draw.circle(self.screen, color_rayo, (ix, iy), 4)

        # Carro orientado
        largo = 1.05 * escala
        ancho_c = 0.54 * escala
        ca = math.cos(estado.angulo)
        sa = math.sin(estado.angulo)
        esquinas = []
        for lx, ly in [(-largo / 2, -ancho_c / 2), (largo / 2, -ancho_c / 2), (largo / 2, ancho_c / 2), (-largo / 2, ancho_c / 2)]:
            wx = car_x + lx * ca - ly * sa
            wy = car_y + lx * sa + ly * ca
            esquinas.append((int(wx), int(wy)))
        pg.draw.polygon(self.screen, self.CAR_COLOR, esquinas)
        nose_x = int(car_x + ca * (largo * 0.45))
        nose_y = int(car_y + sa * (largo * 0.45))
        pg.draw.circle(self.screen, (8, 35, 31), (nose_x, nose_y), 4)

    def _dibujar_icono_accion(self, rect: Any, accion: int) -> None:
        """Dibuja un icono vectorial (flecha / punto / freno) para una acción del historial."""
        pg = self.pygame
        colores = {
            0: self.OK,
            1: self.CAR_COLOR,
            2: self.ACCENT,
            3: self.BAD,
        }
        col = colores.get(int(accion), self.MUTED)
        pg.draw.rect(self.screen, (22, 31, 48), rect, border_radius=6)
        pg.draw.rect(self.screen, col, rect, 1, border_radius=6)

        cx, cy = rect.centerx, rect.centery
        a = int(accion)
        if a == 0:  # Arriba (recto)
            pts = [(cx, cy - 4), (cx - 4, cy + 3), (cx + 4, cy + 3)]
            pg.draw.polygon(self.screen, col, pts)
        elif a == 1:  # Izquierda
            pts = [(cx - 5, cy), (cx + 3, cy - 4), (cx + 3, cy + 4)]
            pg.draw.polygon(self.screen, col, pts)
        elif a == 2:  # Derecha
            pts = [(cx + 5, cy), (cx - 3, cy - 4), (cx - 3, cy + 4)]
            pg.draw.polygon(self.screen, col, pts)
        else:  # 3: Frenar (cuadro rojo)
            pg.draw.rect(self.screen, col, (cx - 3, cy - 3, 6, 6), border_radius=1)

    def _dibujar_panel_lateral(
        self,
        lineas: List[Tuple[str, Tuple[int, int, int]]],
        sensores: List[float],
        velocidad_norm: Optional[float] = None,
        historial: Optional[List[int]] = None,
        accion_texto: Optional[str] = None,
        cp_dist: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> None:
        pg = self.pygame
        x0 = self.AREA_PISTA_W
        w = self.ANCHO - x0
        pg.draw.rect(self.screen, self.PANEL_BG, (x0, 0, w, self.ALTO))
        pg.draw.line(self.screen, self.BORDE, (x0, 0), (x0, self.ALTO), 2)

        y = 18
        for texto, color in lineas:
            surf = self.font_body.render(texto, True, color)
            self.screen.blit(surf, (x0 + 18, y))
            y += 22

        y += 8
        hdr = self.font_h2.render("Variables de entrada", True, self.ACCENT)
        self.screen.blit(hdr, (x0 + 18, y))
        y += 24
        for nombre, val in zip(NOMBRES_RAYOS, sensores):
            lbl = self.font_small.render(f"{nombre:9s}: {val:.2f}", True, self.TEXTO)
            self.screen.blit(lbl, (x0 + 18, y))
            bar_x = x0 + 150
            bar_w = w - 175
            pg.draw.rect(self.screen, (30, 41, 59), (bar_x, y + 2, bar_w, 12), border_radius=4)
            fill_c = self.OK if val > 0.5 else (self.WARN if val > 0.25 else self.BAD)
            pg.draw.rect(self.screen, fill_c, (bar_x, y + 2, int(bar_w * val), 12), border_radius=4)
            y += 21

        if cp_dist is not None:
            if isinstance(cp_dist, (tuple, list)):
                cp_dist_v = limitar(float(cp_dist[0]), 0.0, 1.0)
                cp_ang_v: Optional[float] = limitar(float(cp_dist[1]), -1.0, 1.0)
            else:
                cp_dist_v = limitar(float(cp_dist), 0.0, 1.0)
                cp_ang_v = None
            cp_color = (56, 189, 248)
            lbl = self.font_small.render(f"{'cp_dist':9s}: {cp_dist_v:.2f}", True, cp_color)
            self.screen.blit(lbl, (x0 + 18, y))
            bar_x = x0 + 150
            bar_w = w - 175
            pg.draw.rect(self.screen, (30, 41, 59), (bar_x, y + 2, bar_w, 12), border_radius=4)
            pg.draw.rect(self.screen, cp_color, (bar_x, y + 2, int(bar_w * cp_dist_v), 12), border_radius=4)
            y += 20

            if cp_ang_v is not None:
                lbl_ang = self.font_small.render(f"{'cp_ang':9s}: {cp_ang_v:+.2f}", True, cp_color)
                self.screen.blit(lbl_ang, (x0 + 18, y))
                pg.draw.rect(self.screen, (30, 41, 59), (bar_x, y + 2, bar_w, 12), border_radius=4)
                mid_x = bar_x + bar_w // 2
                dx_bar = int((bar_w // 2) * cp_ang_v)
                if dx_bar >= 0:
                    pg.draw.rect(self.screen, cp_color, (mid_x, y + 2, max(2, dx_bar), 12), border_radius=3)
                else:
                    pg.draw.rect(self.screen, cp_color, (mid_x + dx_bar, y + 2, max(2, -dx_bar), 12), border_radius=3)
                y += 20

        if velocidad_norm is not None:
            v_clamped = limitar(float(velocidad_norm), 0.0, 1.0)
            lbl = self.font_small.render(f"{'velocidad':9s}: {v_clamped:.2f}", True, self.CAR_COLOR)
            self.screen.blit(lbl, (x0 + 18, y))
            bar_x = x0 + 150
            bar_w = w - 175
            pg.draw.rect(self.screen, (30, 41, 59), (bar_x, y + 2, bar_w, 12), border_radius=4)
            pg.draw.rect(self.screen, self.CAR_COLOR, (bar_x, y + 2, int(bar_w * v_clamped), 12), border_radius=4)
            y += 22

        if historial is not None:
            self.screen.blit(self.font_small.render("Historial:", True, self.MUTED), (x0 + 18, y + 1))
            bar_x = x0 + 150
            bar_w = w - 175
            items_h = historial[-5:]
            total_w = len(items_h) * 26 - 4 if items_h else 0
            bx = bar_x + max(0, (bar_w - total_w) // 2)
            for idx_h, act_h in enumerate(items_h):
                box = pg.Rect(bx + idx_h * 26, y, 22, 18)
                self._dibujar_icono_accion(box, act_h)
            y += 30

        if accion_texto is not None:
            self.screen.blit(self.font_h2.render("Salida", True, self.WARN), (x0 + 18, y))
            y += 22
            self.screen.blit(self.font_small.render(f"Accion: {accion_texto}", True, self.TEXTO), (x0 + 18, y))

    def _pedir_nombre_participante(self, permitir_cancelar: bool = False) -> bool:
        """Pide o edita el nombre del participante y actualiza student_id/student_name."""
        pg = self.pygame
        nombre = self.bridge.student_name if self.bridge.student_name != "Oscar_Prueba" else ""
        caja = pg.Rect(260, 340, 700, 48)
        btn_ok = pg.Rect(790, 415, 170, 42)
        btn_cancel = pg.Rect(600, 415, 170, 42)

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE and permitir_cancelar:
                        return True
                    if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        limpio = nombre.strip()
                        if limpio:
                            self.bridge.student_name = limpio
                            self.bridge.student_id = re.sub(r"[^a-zA-Z0-9_-]", "_", limpio).lower()
                            return True
                    elif event.key == pg.K_BACKSPACE:
                        nombre = nombre[:-1]
                    elif event.unicode and event.unicode.isprintable() and len(nombre) < 30:
                        nombre += event.unicode
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if permitir_cancelar and btn_cancel.collidepoint(pos):
                        return True
                    if btn_ok.collidepoint(pos):
                        limpio = nombre.strip()
                        if limpio:
                            self.bridge.student_name = limpio
                            self.bridge.student_id = re.sub(r"[^a-zA-Z0-9_-]", "_", limpio).lower()
                            return True

            self.screen.fill(self.BG)
            self.screen.blit(
                self.font_title.render("UIS Grand Prix Foundationals Models", True, self.ACCENT),
                (260, 220),
            )
            self.screen.blit(
                self.font_h2.render("Ingresa tu nombre de participante:", True, self.TEXTO),
                (260, 290),
            )
            pg.draw.rect(self.screen, self.PANEL_BG, caja, border_radius=10)
            pg.draw.rect(self.screen, self.ACCENT, caja, 2, border_radius=10)
            self.screen.blit(self.font_h2.render(nombre + "|", True, self.TEXTO), (caja.x + 14, caja.y + 12))

            listo = bool(nombre.strip())
            self._dibujar_boton(
                btn_ok,
                "Continuar",
                (15, 52, 48) if listo else (30, 35, 45),
                self.ACCENT if listo else self.BORDE,
            )
            if permitir_cancelar:
                self._dibujar_boton(btn_cancel, "Cancelar", self.PANEL_BG, self.BORDE)
            pg.display.flip()
            self.clock.tick(30)

    def _pantalla_configurar(self) -> None:
        """Pantalla interactiva para configurar ICL, LoRA y Datos sin salir de la app."""
        pg = self.pygame
        btn_volver = pg.Rect(self.ANCHO - 270, 685, 180, 44)

        prec_opts = [20, 50, 100]
        temp_opts = [0.0, 0.2, 0.5, 0.8]
        r_opts = [2, 4, 8, 16, 32]
        alpha_opts = [4, 8, 16, 32, 64]
        lr_opts = [1e-4, 3e-4, 5e-4, 1e-3]
        mods_attn = ["q_proj", "v_proj"]
        mods_full = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        def next_opt(val: Any, opts: List[Any]) -> Any:
            try:
                return opts[(opts.index(val) + 1) % len(opts)]
            except ValueError:
                return opts[0]

        while True:
            # Controles columna izquierda (ICL + Datos)
            btn_icl_k_m = pg.Rect(380, 180, 44, 34)
            btn_icl_k_p = pg.Rect(500, 180, 44, 34)
            btn_icl_sel = pg.Rect(380, 232, 164, 34)
            btn_icl_temp = pg.Rect(380, 284, 164, 34)
            btn_dat_hist_m = pg.Rect(380, 386, 44, 34)
            btn_dat_hist_p = pg.Rect(500, 386, 44, 34)
            btn_dat_prec = pg.Rect(380, 438, 164, 34)

            # Controles columna derecha (LoRA)
            btn_lora_ej_m = pg.Rect(930, 180, 48, 34)
            btn_lora_ej_p = pg.Rect(1062, 180, 48, 34)
            btn_lora_r = pg.Rect(930, 232, 180, 34)
            btn_lora_alpha = pg.Rect(930, 284, 180, 34)
            btn_lora_ep_m = pg.Rect(930, 336, 48, 34)
            btn_lora_ep_p = pg.Rect(1062, 336, 48, 34)
            btn_lora_lr = pg.Rect(930, 388, 180, 34)
            btn_lora_mods = pg.Rect(930, 440, 180, 34)

            # Selector rápido de estrategia arriba
            btn_est_icl = pg.Rect(90, 96, 200, 36)
            btn_est_lora = pg.Rect(305, 96, 200, 36)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    return
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if btn_volver.collidepoint(pos):
                        return
                    if btn_est_icl.collidepoint(pos):
                        cfg.ESTRATEGIA = "ICL"
                    elif btn_est_lora.collidepoint(pos):
                        cfg.ESTRATEGIA = "LORA"
                    elif btn_icl_k_m.collidepoint(pos):
                        cfg.CONFIG_ICL["k_shots"] = max(1, int(cfg.CONFIG_ICL["k_shots"]) - 1)
                    elif btn_icl_k_p.collidepoint(pos):
                        cfg.CONFIG_ICL["k_shots"] = min(30, int(cfg.CONFIG_ICL["k_shots"]) + 1)
                    elif btn_icl_sel.collidepoint(pos):
                        cfg.CONFIG_ICL["seleccion"] = "random" if cfg.CONFIG_ICL["seleccion"] == "knn" else "knn"
                    elif btn_icl_temp.collidepoint(pos):
                        cfg.CONFIG_ICL["temperature"] = next_opt(float(cfg.CONFIG_ICL["temperature"]), temp_opts)
                    elif btn_dat_hist_m.collidepoint(pos):
                        cfg.CONFIG_DATOS["ventana_historia"] = max(1, int(cfg.CONFIG_DATOS["ventana_historia"]) - 1)
                    elif btn_dat_hist_p.collidepoint(pos):
                        cfg.CONFIG_DATOS["ventana_historia"] = min(8, int(cfg.CONFIG_DATOS["ventana_historia"]) + 1)
                    elif btn_dat_prec.collidepoint(pos):
                        cfg.CONFIG_DATOS["precision_sensores"] = next_opt(int(cfg.CONFIG_DATOS["precision_sensores"]), prec_opts)
                    elif btn_lora_ej_m.collidepoint(pos):
                        cfg.CONFIG_LORA["num_ejemplos"] = max(20, int(cfg.CONFIG_LORA["num_ejemplos"]) - 20)
                    elif btn_lora_ej_p.collidepoint(pos):
                        cfg.CONFIG_LORA["num_ejemplos"] = min(1000, int(cfg.CONFIG_LORA["num_ejemplos"]) + 20)
                    elif btn_lora_r.collidepoint(pos):
                        cfg.CONFIG_LORA["lora_r"] = next_opt(int(cfg.CONFIG_LORA["lora_r"]), r_opts)
                    elif btn_lora_alpha.collidepoint(pos):
                        cfg.CONFIG_LORA["lora_alpha"] = next_opt(int(cfg.CONFIG_LORA["lora_alpha"]), alpha_opts)
                    elif btn_lora_ep_m.collidepoint(pos):
                        cfg.CONFIG_LORA["epochs"] = max(1, int(cfg.CONFIG_LORA["epochs"]) - 1)
                    elif btn_lora_ep_p.collidepoint(pos):
                        cfg.CONFIG_LORA["epochs"] = min(15, int(cfg.CONFIG_LORA["epochs"]) + 1)
                    elif btn_lora_lr.collidepoint(pos):
                        cfg.CONFIG_LORA["learning_rate"] = next_opt(float(cfg.CONFIG_LORA["learning_rate"]), lr_opts)
                    elif btn_lora_mods.collidepoint(pos):
                        es_full = len(cfg.CONFIG_LORA["target_modules"]) > 2
                        cfg.CONFIG_LORA["target_modules"] = list(mods_attn if es_full else mods_full)

            self.screen.fill(self.BG)
            self.screen.blit(
                self.font_title.render("Configurar Estrategias (ICL y LoRA)", True, self.ACCENT),
                (90, 42),
            )
            self._dibujar_boton(
                btn_est_icl,
                "Estrategia: ICL",
                (15, 52, 48) if cfg.ESTRATEGIA == "ICL" else self.PANEL_BG,
                self.ACCENT if cfg.ESTRATEGIA == "ICL" else self.BORDE,
            )
            self._dibujar_boton(
                btn_est_lora,
                "Estrategia: LORA",
                (35, 36, 78) if cfg.ESTRATEGIA == "LORA" else self.PANEL_BG,
                (129, 140, 248) if cfg.ESTRATEGIA == "LORA" else self.BORDE,
            )

            usa_icl = cfg.ESTRATEGIA in {"ICL", "LORA_ICL"}
            usa_lora = cfg.ESTRATEGIA in {"LORA", "LORA_ICL"}

            # Tarjeta Izquierda: ICL y Datos
            pg.draw.rect(self.screen, self.PANEL_BG, pg.Rect(90, 148, 490, 360), border_radius=12)
            pg.draw.rect(self.screen, self.ACCENT if usa_icl else self.BORDE, pg.Rect(90, 148, 490, 360), 2, border_radius=12)
            icl_lbl = "ICL (Prompt) [ACTIVO]" if usa_icl else "ICL (Prompt) [INACTIVO]"
            self.screen.blit(self.font_h2.render(icl_lbl, True, self.ACCENT if usa_icl else self.MUTED), (115, 154))
            self.screen.blit(self.font_body.render("k_shots (1..30):", True, self.TEXTO), (115, 187))
            self._dibujar_boton(btn_icl_k_m, "-", self.BG, self.BORDE)
            k_surf = self.font_body.render(str(cfg.CONFIG_ICL["k_shots"]), True, self.TEXTO)
            self.screen.blit(k_surf, (462 - k_surf.get_width() // 2, 187))
            self._dibujar_boton(btn_icl_k_p, "+", self.BG, self.BORDE)

            self.screen.blit(self.font_body.render("Seleccion:", True, self.TEXTO), (115, 239))
            self._dibujar_boton(btn_icl_sel, str(cfg.CONFIG_ICL["seleccion"]).upper(), self.BG, self.ACCENT)

            self.screen.blit(self.font_body.render("Temperature:", True, self.TEXTO), (115, 291))
            self._dibujar_boton(btn_icl_temp, f"{cfg.CONFIG_ICL['temperature']:.1f}", self.BG, self.BORDE)

            self.screen.blit(self.font_h2.render("Datos (Compartido)", True, self.WARN), (115, 348))
            self.screen.blit(self.font_body.render("Ventana historial:", True, self.TEXTO), (115, 393))
            self._dibujar_boton(btn_dat_hist_m, "-", self.BG, self.BORDE)
            h_surf = self.font_body.render(str(cfg.CONFIG_DATOS["ventana_historia"]), True, self.TEXTO)
            self.screen.blit(h_surf, (462 - h_surf.get_width() // 2, 393))
            self._dibujar_boton(btn_dat_hist_p, "+", self.BG, self.BORDE)

            self.screen.blit(self.font_body.render("Precision sensores:", True, self.TEXTO), (115, 445))
            self._dibujar_boton(btn_dat_prec, f"{cfg.CONFIG_DATOS['precision_sensores']} bins", self.BG, self.BORDE)

            # Tarjeta Derecha: LoRA
            pg.draw.rect(self.screen, self.PANEL_BG, pg.Rect(630, 148, 500, 360), border_radius=12)
            pg.draw.rect(self.screen, (129, 140, 248) if usa_lora else self.BORDE, pg.Rect(630, 148, 500, 360), 2, border_radius=12)
            lora_lbl = "LoRA (Pesos PEFT) [ACTIVO]" if usa_lora else "LoRA (Pesos PEFT) [INACTIVO]"
            self.screen.blit(self.font_h2.render(lora_lbl, True, (165, 180, 252) if usa_lora else self.MUTED), (655, 154))

            self.screen.blit(self.font_body.render("Num ejemplos:", True, self.TEXTO), (655, 187))
            self._dibujar_boton(btn_lora_ej_m, "-20", self.BG, self.BORDE)
            ej_surf = self.font_body.render(str(cfg.CONFIG_LORA["num_ejemplos"]), True, self.TEXTO)
            self.screen.blit(ej_surf, (1020 - ej_surf.get_width() // 2, 187))
            self._dibujar_boton(btn_lora_ej_p, "+20", self.BG, self.BORDE)

            self.screen.blit(self.font_body.render("lora_r (rango):", True, self.TEXTO), (655, 239))
            self._dibujar_boton(btn_lora_r, f"r = {cfg.CONFIG_LORA['lora_r']}", self.BG, self.BORDE)

            self.screen.blit(self.font_body.render("lora_alpha:", True, self.TEXTO), (655, 291))
            self._dibujar_boton(btn_lora_alpha, f"alpha = {cfg.CONFIG_LORA['lora_alpha']}", self.BG, self.BORDE)

            self.screen.blit(self.font_body.render("Epochs (1..15):", True, self.TEXTO), (655, 343))
            self._dibujar_boton(btn_lora_ep_m, "-", self.BG, self.BORDE)
            ep_surf = self.font_body.render(str(cfg.CONFIG_LORA["epochs"]), True, self.TEXTO)
            self.screen.blit(ep_surf, (1020 - ep_surf.get_width() // 2, 343))
            self._dibujar_boton(btn_lora_ep_p, "+", self.BG, self.BORDE)

            self.screen.blit(self.font_body.render("Learning rate:", True, self.TEXTO), (655, 395))
            self._dibujar_boton(btn_lora_lr, f"{cfg.CONFIG_LORA['learning_rate']:.0e}", self.BG, self.BORDE)

            self.screen.blit(self.font_body.render("Capas LoRA:", True, self.TEXTO), (655, 447))
            lbl_mods = "Atencion + MLP" if len(cfg.CONFIG_LORA["target_modules"]) > 2 else "Solo Atencion (q,v)"
            self._dibujar_boton(btn_lora_mods, lbl_mods, self.BG, (129, 140, 248))

            self.screen.blit(
                self.font_small.render("Nota: puedes usar ICL y LORA alternando la estrategia arriba y comparando ambos en el ranking.", True, self.MUTED),
                (90, 540),
            )
            self._dibujar_boton(btn_volver, "Volver al Menu", (15, 52, 48), self.ACCENT)
            pg.display.flip()
            self.clock.tick(30)

    def _pantalla_estrategia(self, estrategia_sel: str) -> None:
        """Pantalla dedicada para configurar, entrenar/elegir LoRA y lanzar simulaciones de la estrategia."""
        pg = self.pygame
        cfg.ESTRATEGIA = estrategia_sel
        usa_icl = estrategia_sel in {"ICL", "LORA_ICL"}
        usa_lora = estrategia_sel in {"LORA", "LORA_ICL"}
        if usa_lora:
            self.loras_disponibles = self.bridge.listar_adaptadores()
            if self.loras_disponibles and self.lora_activo not in self.loras_disponibles:
                self.lora_activo = self.loras_disponibles[-1]

        prec_opts = [20, 50, 100]
        temp_opts = [0.0, 0.2, 0.5, 0.8]
        mods_attn = ["q_proj", "v_proj"]
        mods_full = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        all_meta: Dict[str, Tuple[str, str, str]] = {
            "k_shots": (
                "k_shots (Ejemplos en prompt)",
                "Cantidad de jugadas similares de mis_jugadas.json que se incluyen en el prompt como contexto para decidir la accion.",
                "Restriccion: Entero entre 1 y 30. Mas ejemplos dan mas contexto, pero aumentan la latencia.",
            ),
            "seleccion": (
                "Seleccion de ejemplos",
                "KNN busca los k_shots estados mas parecidos (por sensores y angulo al checkpoint). RANDOM toma ejemplos al azar.",
                "Restriccion: Solo valores 'KNN' o 'RANDOM'.",
            ),
            "temperature": (
                "Temperature",
                "Controla la aleatoriedad al elegir la accion. En 0.0 siempre elige la opcion mas probable (determinista).",
                "Restriccion: Decimal en {0.0, 0.2, 0.5, 0.8} (rango 0.0 a 2.0).",
            ),
            "ventana_historia": (
                "Ventana historial",
                "Numero de acciones previas (H) que ve el modelo para dar continuidad al giro sin zigzaguear.",
                "Restriccion: Entero entre 1 y 8 frames.",
            ),
            "precision_sensores": (
                "Precision sensores",
                "Numero de intervalos (bins) usados para discretizar las distancias [0..1] a enteros en el prompt.",
                "Restriccion: Entero discreto entre {20, 50, 100} bins (servidor admite 5 a 500).",
            ),
            "num_ejemplos": (
                "Num ejemplos",
                "Cantidad maxima de jugadas tomadas de mis_jugadas.json para entrenar el adaptador LoRA.",
                "Restriccion: Entero entre 1 y 300 muestras (tope del servidor para mantener velocidad).",
            ),
            "lora_r": (
                "lora_r (Rango)",
                "Rango de las matrices de adaptacion LoRA. Valores altos aprenden patrones mas complejos; bajos evitan sobreajuste.",
                "Restriccion: Entero entre 1 y 32. Recomendado potencia de 2 (2, 4, 8, 16, 32).",
            ),
            "lora_alpha": (
                "lora_alpha (Escala)",
                "Factor de escala de los pesos LoRA respecto al modelo base Qwen2.5.",
                "Restriccion: Entero entre 1 y 128. Regla tipica: multiplo de lora_r (ej. 2 x lora_r).",
            ),
            "epochs": (
                "Epocas",
                "Numero de pasadas completas sobre el dataset durante el entrenamiento.",
                "Restriccion: Entero entre 1 y 15 epocas (tope del servidor).",
            ),
            "learning_rate": (
                "Learning rate",
                "Tasa de aprendizaje del optimizador AdamW (ej. 3e-4 = 0.0003). Muy alto inestabiliza; muy bajo no aprende.",
                "Restriccion: Decimal entre 1e-5 (0.00001) y 0.05.",
            ),
            "target_modules": (
                "Capas LoRA",
                "Define que matrices del Transformer se adaptan: Solo Atencion (q_proj, v_proj) o Atencion + MLP (7 capas).",
                "Restriccion: Elegir entre 'Solo Atencion' (2 modulos) o 'Atencion + MLP' (7 modulos).",
            ),
        }

        lora_limits: Dict[str, Tuple[float, float, bool]] = {
            "num_ejemplos": (1, 300, True),
            "lora_r": (1, 32, True),
            "lora_alpha": (1, 128, True),
            "epochs": (1, 15, True),
            "learning_rate": (1e-5, 0.05, False),
        }

        campo_activo: Optional[str] = None
        texto_edit: str = ""
        info_abierta: Optional[str] = None

        def _commit_campo() -> None:
            nonlocal campo_activo, texto_edit
            if not campo_activo or campo_activo not in lora_limits:
                campo_activo = None
                texto_edit = ""
                return
            vmin, vmax, es_int = lora_limits[campo_activo]
            try:
                val = float(texto_edit.strip().replace(",", "."))
                val = max(vmin, min(vmax, val))
                cfg.CONFIG_LORA[campo_activo] = int(round(val)) if es_int else float(val)
            except ValueError:
                pass
            campo_activo = None
            texto_edit = ""

        def next_opt(val: Any, opts: List[Any]) -> Any:
            try:
                return opts[(opts.index(val) + 1) % len(opts)]
            except ValueError:
                return opts[0]

        def _iniciar_entrenamiento_bg() -> None:
            with self._train_lock:
                if self._train_state["running"]:
                    self.estado_menu_msg = "Ya hay un entrenamiento LoRA en curso."
                    return
            jugadas = cargar_jugadas()
            if not jugadas:
                self.estado_menu_msg = "Error: no hay jugadas en data/mis_jugadas.json. Graba primero en [G]."
                return
            existentes = self.bridge.listar_adaptadores()
            if len(existentes) >= 3:
                a_borrar = self._elegir_lora_a_sobrescribir(existentes[:3])
                if not a_borrar:
                    self.estado_menu_msg = "Entrenamiento LoRA cancelado."
                    return
                self.bridge.eliminar_adaptador(a_borrar)
                existentes = self.bridge.listar_adaptadores()

            nuevo_nombre = f"v{len(existentes) + 1}_{time.strftime('%H%M%S')}"
            self._train_cancel_ev.clear()
            with self._train_lock:
                self._train_state = {
                    "running": True,
                    "name": nuevo_nombre,
                    "pct": 0.0,
                    "epoch": 0,
                    "total_epochs": int(cfg.CONFIG_LORA.get("epochs", 8)),
                    "loss": None,
                    "t_start": time.monotonic(),
                }
            self.estado_menu_msg = f"Entrenando '{nuevo_nombre}' en segundo plano..."

            lora_snap = copy.deepcopy(cfg.CONFIG_LORA)
            datos_snap = copy.deepcopy(cfg.CONFIG_DATOS)

            def _on_prog(job: Dict[str, Any]) -> None:
                with self._train_lock:
                    self._train_state["pct"] = float(job.get("pct", self._train_state["pct"]))
                    self._train_state["epoch"] = int(job.get("epoch", self._train_state["epoch"]))
                    self._train_state["total_epochs"] = int(job.get("total_epochs", self._train_state["total_epochs"]))
                    self._train_state["loss"] = job.get("loss")

            def _worker() -> None:
                try:
                    res = self.bridge.entrenar_lora(
                        jugadas,
                        lora_snap,
                        datos_snap,
                        adapter_name=nuevo_nombre,
                        on_progress=_on_prog,
                        cancel_event=self._train_cancel_ev,
                    )
                    self.loras_disponibles = self.bridge.listar_adaptadores()
                    self.lora_activo = nuevo_nombre
                    self.estado_menu_msg = (
                        f"LoRA '{nuevo_nombre}' OK ({res['ejemplos_entrenados']} ej, "
                        f"loss={res['loss_final']}, t={res['tiempo_entrenamiento_s']}s)."
                    )
                except InterruptedError:
                    self.estado_menu_msg = f"Entrenamiento '{nuevo_nombre}' cancelado."
                except Exception as exc:
                    self.estado_menu_msg = f"Error LoRA: {exc}"
                finally:
                    with self._train_lock:
                        self._train_state["running"] = False

            threading.Thread(target=_worker, daemon=True).start()

        while True:
            # Controles ICL + Datos (Izquierda) con botones [i]
            info_icl_k = pg.Rect(344, 170, 28, 28)
            btn_icl_k_m = pg.Rect(382, 168, 44, 32)
            btn_icl_k_p = pg.Rect(502, 168, 44, 32)
            info_icl_sel = pg.Rect(344, 216, 28, 28)
            btn_icl_sel = pg.Rect(382, 214, 164, 32)
            info_icl_temp = pg.Rect(344, 262, 28, 28)
            btn_icl_temp = pg.Rect(382, 260, 164, 32)

            info_dat_hist = pg.Rect(344, 346, 28, 28)
            btn_dat_hist_m = pg.Rect(382, 344, 44, 32)
            btn_dat_hist_p = pg.Rect(502, 344, 44, 32)
            info_dat_prec = pg.Rect(344, 392, 28, 28)
            btn_dat_prec = pg.Rect(382, 390, 164, 32)

            # Controles LoRA (Derecha): input directo + boton [i] de info
            lora_rows = [
                ("num_ejemplos", 172),
                ("lora_r", 214),
                ("lora_alpha", 256),
                ("epochs", 298),
                ("learning_rate", 340),
                ("target_modules", 382),
            ]
            rects_info = {k: pg.Rect(885, y + 2, 28, 28) for k, y in lora_rows}
            rects_input = {k: pg.Rect(925, y, 185, 32) for k, y in lora_rows}
            rects_info["k_shots"] = info_icl_k
            rects_info["seleccion"] = info_icl_sel
            rects_info["temperature"] = info_icl_temp
            rects_info["ventana_historia"] = info_dat_hist
            rects_info["precision_sensores"] = info_dat_prec

            # Selector de adaptador LoRA entrenado (hasta 3)
            btns_lora_ver = [
                pg.Rect(90 + i * 210, 488, 195, 36)
                for i in range(len(self.loras_disponibles[:3]))
            ]

            # Botones de acción abajo
            btn_entrenar = pg.Rect(90, 555, 250, 46)
            btn_simular = pg.Rect(360 if usa_lora else 90, 555, 280, 46)
            btn_evaluar = pg.Rect(660 if usa_lora else 390, 555, 290, 46)
            btn_volver = pg.Rect(self.ANCHO - 260, 555, 170, 46)
            btn_cancel_train = pg.Rect(self.ANCHO - 260, 624, 170, 34)
            btn_cerrar_info = pg.Rect(self.ANCHO // 2 - 90, 445, 180, 38)

            with self._train_lock:
                tr_snap = dict(self._train_state)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    _commit_campo()
                    return
                if event.type == pg.KEYDOWN:
                    if info_abierta is not None:
                        if event.key in (pg.K_ESCAPE, pg.K_RETURN, pg.K_KP_ENTER):
                            info_abierta = None
                        continue
                    if campo_activo is not None:
                        if event.key in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_TAB):
                            _commit_campo()
                        elif event.key == pg.K_ESCAPE:
                            campo_activo = None
                            texto_edit = ""
                        elif event.key == pg.K_BACKSPACE:
                            texto_edit = texto_edit[:-1]
                        elif event.unicode and event.unicode in "0123456789.eE-+":
                            if len(texto_edit) < 14:
                                texto_edit += event.unicode
                        continue
                    if event.key == pg.K_ESCAPE:
                        return
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if info_abierta is not None:
                        if btn_cerrar_info.collidepoint(pos) or not pg.Rect(220, 200, self.ANCHO - 440, 300).collidepoint(pos):
                            info_abierta = None
                        continue

                    if usa_lora:
                        clic_en_input = False
                        for k_inp, r_inp in rects_input.items():
                            if k_inp != "target_modules" and r_inp.collidepoint(pos):
                                if campo_activo != k_inp:
                                    _commit_campo()
                                    campo_activo = k_inp
                                    v_cur = cfg.CONFIG_LORA[k_inp]
                                    texto_edit = f"{v_cur:.0e}" if k_inp == "learning_rate" else str(v_cur)
                                clic_en_input = True
                                break
                        if clic_en_input:
                            continue

                    # Botones [i] para TODOS los parámetros visibles
                    clic_en_info = False
                    for k_inf, r_inf in rects_info.items():
                        if k_inf in {"k_shots", "seleccion", "temperature"} and not usa_icl:
                            continue
                        if k_inf in lora_limits and not usa_lora:
                            continue
                        if k_inf == "target_modules" and not usa_lora:
                            continue
                        if r_inf.collidepoint(pos):
                            _commit_campo()
                            info_abierta = k_inf
                            clic_en_info = True
                            break
                    if clic_en_info:
                        continue

                    _commit_campo()
                    if btn_volver.collidepoint(pos):
                        return
                    if tr_snap["running"] and btn_cancel_train.collidepoint(pos):
                        self._train_cancel_ev.set()
                        self.estado_menu_msg = "Cancelando entrenamiento..."
                        continue
                    if btn_dat_hist_m.collidepoint(pos):
                        cfg.CONFIG_DATOS["ventana_historia"] = max(1, int(cfg.CONFIG_DATOS["ventana_historia"]) - 1)
                    elif btn_dat_hist_p.collidepoint(pos):
                        cfg.CONFIG_DATOS["ventana_historia"] = min(8, int(cfg.CONFIG_DATOS["ventana_historia"]) + 1)
                    elif btn_dat_prec.collidepoint(pos):
                        cfg.CONFIG_DATOS["precision_sensores"] = next_opt(cfg.CONFIG_DATOS["precision_sensores"], prec_opts)
                    elif usa_icl and btn_icl_k_m.collidepoint(pos):
                        cfg.CONFIG_ICL["k_shots"] = max(1, int(cfg.CONFIG_ICL["k_shots"]) - 1)
                    elif usa_icl and btn_icl_k_p.collidepoint(pos):
                        cfg.CONFIG_ICL["k_shots"] = min(30, int(cfg.CONFIG_ICL["k_shots"]) + 1)
                    elif usa_icl and btn_icl_sel.collidepoint(pos):
                        cfg.CONFIG_ICL["seleccion"] = "random" if cfg.CONFIG_ICL["seleccion"] == "knn" else "knn"
                    elif usa_icl and btn_icl_temp.collidepoint(pos):
                        cfg.CONFIG_ICL["temperature"] = next_opt(cfg.CONFIG_ICL["temperature"], temp_opts)
                    elif usa_lora and rects_input["target_modules"].collidepoint(pos):
                        cur = cfg.CONFIG_LORA["target_modules"]
                        cfg.CONFIG_LORA["target_modules"] = mods_attn if len(cur) > 2 else mods_full
                    elif usa_lora and btn_entrenar.collidepoint(pos):
                        _iniciar_entrenamiento_bg()
                    elif btn_simular.collidepoint(pos):
                        if usa_lora and not self.loras_disponibles:
                            self.estado_menu_msg = "Primero entrena al menos 1 modelo LoRA antes de simular."
                        else:
                            self._modo_prueba_a()
                    elif btn_evaluar.collidepoint(pos):
                        if usa_lora and not self.loras_disponibles:
                            self.estado_menu_msg = "Primero entrena al menos 1 modelo LoRA antes de evaluar."
                        else:
                            self._modo_evaluacion_b()
                    elif usa_lora:
                        for idx_v, b_v in enumerate(btns_lora_ver):
                            if b_v.collidepoint(pos):
                                self.lora_activo = self.loras_disponibles[idx_v]
                                self.estado_menu_msg = f"Configuracion seleccionada para simular: {self.lora_activo}"
                                break

            self.screen.fill(self.BG)
            tit_map = {
                "LORA": "Estrategia: LORA (Fine-Tuning PEFT)",
                "ICL": "Estrategia: ICL (In-Context Learning)",
                "LORA_ICL": "Estrategia: LORA + ICL (Hibrido)",
            }
            self.screen.blit(self.font_title.render(tit_map.get(estrategia_sel, estrategia_sel), True, self.ACCENT), (90, 40))
            self.screen.blit(
                self.font_body.render(f"Participante: {self.bridge.student_name} | Ajusta parametros, entrena/elige modelo y simula.", True, self.MUTED),
                (90, 82),
            )

            # Tarjeta Izquierda: ICL (si aplica) + Datos
            pg.draw.rect(self.screen, self.PANEL_BG, pg.Rect(90, 128, 490, 310), border_radius=12)
            pg.draw.rect(self.screen, self.ACCENT if usa_icl else self.BORDE, pg.Rect(90, 128, 490, 310), 2, border_radius=12)
            if usa_icl:
                self.screen.blit(self.font_h2.render("Parametros ICL (Prompt)", True, self.ACCENT), (115, 136))
                self.screen.blit(self.font_body.render("k_shots (1..30):", True, self.TEXTO), (115, 174))
                self._dibujar_boton(info_icl_k, "i", (24, 38, 62), self.ACCENT)
                self._dibujar_boton(btn_icl_k_m, "-", self.BG, self.BORDE)
                ks = self.font_body.render(str(cfg.CONFIG_ICL["k_shots"]), True, self.TEXTO)
                self.screen.blit(ks, (464 - ks.get_width() // 2, 174))
                self._dibujar_boton(btn_icl_k_p, "+", self.BG, self.BORDE)

                self.screen.blit(self.font_body.render("Seleccion:", True, self.TEXTO), (115, 220))
                self._dibujar_boton(info_icl_sel, "i", (24, 38, 62), self.ACCENT)
                self._dibujar_boton(btn_icl_sel, str(cfg.CONFIG_ICL["seleccion"]).upper(), self.BG, self.ACCENT)

                self.screen.blit(self.font_body.render("Temperature:", True, self.TEXTO), (115, 266))
                self._dibujar_boton(info_icl_temp, "i", (24, 38, 62), self.ACCENT)
                self._dibujar_boton(btn_icl_temp, f"{cfg.CONFIG_ICL['temperature']:.1f}", self.BG, self.BORDE)
            else:
                self.screen.blit(self.font_h2.render("ICL desactivado en modo LORA puro", True, self.MUTED), (115, 150))
                self.screen.blit(self.font_small.render("En LORA puro el modelo responde zero-shot (0 ejemplos).", True, self.MUTED), (115, 185))

            self.screen.blit(self.font_h2.render("Representacion de Datos", True, self.WARN), (115, 312))
            self.screen.blit(self.font_body.render("Ventana historial:", True, self.TEXTO), (115, 350))
            self._dibujar_boton(info_dat_hist, "i", (24, 38, 62), self.ACCENT)
            self._dibujar_boton(btn_dat_hist_m, "-", self.BG, self.BORDE)
            hs = self.font_body.render(str(cfg.CONFIG_DATOS["ventana_historia"]), True, self.TEXTO)
            self.screen.blit(hs, (464 - hs.get_width() // 2, 350))
            self._dibujar_boton(btn_dat_hist_p, "+", self.BG, self.BORDE)
            self.screen.blit(self.font_body.render("Precision sensores:", True, self.TEXTO), (115, 396))
            self._dibujar_boton(info_dat_prec, "i", (24, 38, 62), self.ACCENT)
            self._dibujar_boton(btn_dat_prec, f"{cfg.CONFIG_DATOS['precision_sensores']} bins", self.BG, self.BORDE)

            # Tarjeta Derecha: LoRA con inputs directos y botones [i]
            pg.draw.rect(self.screen, self.PANEL_BG, pg.Rect(630, 128, 500, 310), border_radius=12)
            pg.draw.rect(self.screen, (129, 140, 248) if usa_lora else self.BORDE, pg.Rect(630, 128, 500, 310), 2, border_radius=12)
            if usa_lora:
                self.screen.blit(self.font_h2.render("Parametros LoRA (Clic para editar)", True, (165, 180, 252)), (655, 136))
                for k_row, y_row in lora_rows:
                    tit_r = all_meta[k_row][0].split(" (")[0] + ":"
                    self.screen.blit(self.font_body.render(tit_r, True, self.TEXTO), (655, y_row + 5))
                    self._dibujar_boton(rects_info[k_row], "i", (24, 38, 62), self.ACCENT)
                    r_in = rects_input[k_row]
                    if k_row == "target_modules":
                        lbl_mods = "Atencion + MLP" if len(cfg.CONFIG_LORA["target_modules"]) > 2 else "Solo Atencion"
                        self._dibujar_boton(r_in, lbl_mods, self.BG, (129, 140, 248))
                    else:
                        editando = (campo_activo == k_row)
                        pg.draw.rect(self.screen, (15, 23, 42) if editando else self.BG, r_in, border_radius=6)
                        pg.draw.rect(self.screen, self.ACCENT if editando else self.BORDE, r_in, 2, border_radius=6)
                        if editando:
                            txt_show = texto_edit + "|"
                        elif k_row == "learning_rate":
                            txt_show = f"{cfg.CONFIG_LORA[k_row]:.0e}"
                        else:
                            txt_show = str(cfg.CONFIG_LORA[k_row])
                        s_val = self.font_body.render(txt_show, True, self.TEXTO)
                        self.screen.blit(s_val, (r_in.x + 12, r_in.y + 7))
            else:
                self.screen.blit(self.font_h2.render("LoRA no se usa en modo ICL puro", True, self.MUTED), (655, 150))
                self.screen.blit(self.font_small.render("ICL usa el modelo base congelado sin entrenar pesos.", True, self.MUTED), (655, 185))

            # Selector de configuración/modelo LoRA entrenado para simular
            if usa_lora:
                self.screen.blit(
                    self.font_body.render("Modelo LoRA entrenado a simular (max 3):", True, self.TEXTO),
                    (90, 456),
                )
                if not self.loras_disponibles:
                    self.screen.blit(
                        self.font_small.render("Aun no tienes modelos LoRA entrenados. Pulsa 'Entrenar Nuevo LoRA' abajo.", True, self.WARN),
                        (90, 494),
                    )
                else:
                    for idx_v, b_v in enumerate(btns_lora_ver):
                        nm_v = self.loras_disponibles[idx_v]
                        activo = (nm_v == self.lora_activo)
                        self._dibujar_boton(
                            b_v,
                            f"{'● ' if activo else ''}{nm_v}",
                            (35, 36, 78) if activo else self.PANEL_BG,
                            (129, 140, 248) if activo else self.BORDE,
                        )
                self._dibujar_boton(
                    btn_entrenar,
                    "Entrenando..." if tr_snap["running"] else "Entrenar Nuevo LoRA",
                    (35, 36, 78),
                    (129, 140, 248),
                )

            self._dibujar_boton(btn_simular, "Simular Prueba (P1-P3)", (15, 52, 48), self.ACCENT)
            self._dibujar_boton(btn_evaluar, "Evaluacion Oficial (B1-B3)", (52, 36, 16), self.WARN)
            self._dibujar_boton(btn_volver, "Volver al Menu", self.PANEL_BG, self.BORDE)

            # Barra de progreso de entrenamiento NO bloqueante debajo de los botones
            if tr_snap["running"]:
                barra_tr = pg.Rect(90, 626, self.ANCHO - 370, 26)
                pg.draw.rect(self.screen, self.PANEL_BG, barra_tr, border_radius=6)
                pg.draw.rect(self.screen, (129, 140, 248), barra_tr, 2, border_radius=6)
                fw = max(0, int((barra_tr.w - 4) * (float(tr_snap["pct"]) / 100.0)))
                if fw > 0:
                    pg.draw.rect(
                        self.screen,
                        (129, 140, 248),
                        pg.Rect(barra_tr.x + 2, barra_tr.y + 2, fw, barra_tr.h - 4),
                        border_radius=4,
                    )
                loss_t = f"{tr_snap['loss']:.4f}" if isinstance(tr_snap["loss"], (int, float)) else "..."
                elap = time.monotonic() - float(tr_snap["t_start"])
                lbl_tr = (
                    f"Entrenando '{tr_snap['name']}': {tr_snap['pct']:.0f}% | "
                    f"Epoca {tr_snap['epoch']}/{tr_snap['total_epochs']} | Loss {loss_t} | {elap:.0f}s"
                )
                self.screen.blit(self.font_small.render(lbl_tr, True, self.TEXTO), (90, 660))
                self._dibujar_boton(
                    btn_cancel_train,
                    "Cancelando..." if self._train_cancel_ev.is_set() else "Cancelar",
                    (56, 20, 26),
                    self.BAD,
                )
            else:
                self.screen.blit(self.font_body.render(f"Estado: {self.estado_menu_msg}", True, self.TEXTO), (90, 634))

            # Popup de información [i] para cualquier parámetro
            if info_abierta is not None and info_abierta in all_meta:
                tit_i, desc_i, rest_i = all_meta[info_abierta]
                overlay = pg.Surface((self.ANCHO, self.ALTO), pg.SRCALPHA)
                overlay.fill((0, 0, 0, 165))
                self.screen.blit(overlay, (0, 0))
                modal = pg.Rect(220, 200, self.ANCHO - 440, 300)
                pg.draw.rect(self.screen, self.PANEL_BG, modal, border_radius=14)
                pg.draw.rect(self.screen, self.ACCENT, modal, 2, border_radius=14)
                self.screen.blit(self.font_title.render(f"Info: {tit_i}", True, self.ACCENT), (modal.x + 28, modal.y + 24))

                def _draw_wrapped(text_in: str, y_start: int, color: Tuple[int, int, int]) -> int:
                    lineas_w: List[str] = []
                    cur_l = ""
                    for pal in text_in.split():
                        if len(cur_l) + len(pal) + 1 <= 68:
                            cur_l = (cur_l + " " + pal).strip()
                        else:
                            lineas_w.append(cur_l)
                            cur_l = pal
                    if cur_l:
                        lineas_w.append(cur_l)
                    yy = y_start
                    for ln_w in lineas_w[:3]:
                        self.screen.blit(self.font_body.render(ln_w, True, color), (modal.x + 28, yy))
                        yy += 24
                    return yy

                y_next = _draw_wrapped(desc_i, modal.y + 82, self.TEXTO)
                _draw_wrapped(rest_i, max(y_next + 10, modal.y + 168), self.WARN)
                self._dibujar_boton(btn_cerrar_info, "Entendido", (15, 52, 48), self.ACCENT)

            pg.display.flip()
            self.clock.tick(30)
            pg.display.flip()
            self.clock.tick(30)

    def _pantalla_menu(self) -> Optional[int]:
        pg = self.pygame
        tarjetas_estrategia = [
            ("LORA", "LORA", "Ajusta hiperparametros, entrena hasta 3 modelos y simula zero-shot", (35, 36, 78), (129, 140, 248)),
            ("ICL", "ICL", "In-Context Learning: ajusta k_shots y KNN sin entrenar pesos", (15, 52, 48), self.ACCENT),
            ("LORA_ICL", "LORA + ICL", "Hibrido: usa tu modelo LoRA entrenado + ejemplos en contexto", (52, 36, 16), self.WARN),
        ]
        btn_carrera_final = pg.Rect(self.ANCHO - 495, 82, 240, 32)
        btn_editar_nombre = pg.Rect(self.ANCHO - 240, 82, 150, 32)
        btn_grabar = pg.Rect(90, 205, self.ANCHO - 180, 88)
        btn_copiar_cmd = pg.Rect(self.ANCHO - 250, 142, 160, 28)
        btn_cola_abrir = pg.Rect(self.ANCHO - 375, 626, 140, 28)
        btn_cola_cancel = pg.Rect(self.ANCHO - 225, 626, 135, 28)
        cmd_backend = f'cd /d "{BASE_DIR.parent / "backend_server"}" && python -m uvicorn server:app --host 0.0.0.0 --port 8000'

        card_w = (self.ANCHO - 180 - 40) // 3
        card_rects = [pg.Rect(90 + i * (card_w + 20), 350, card_w, 165) for i in range(3)]

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.KEYDOWN:
                    if event.key in (pg.K_1, pg.K_KP1):
                        self._pantalla_estrategia("LORA")
                    if event.key in (pg.K_2, pg.K_KP2):
                        self._pantalla_estrategia("ICL")
                    if event.key in (pg.K_3, pg.K_KP3):
                        self._pantalla_estrategia("LORA_ICL")
                    if event.key == pg.K_g:
                        return 1
                    if event.key == pg.K_c:
                        self._pantalla_carrera_todos()
                    if event.key == pg.K_h:
                        self._chequear_salud_silenciosa()
                    if event.key == pg.K_ESCAPE:
                        return None
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if btn_editar_nombre.collidepoint((mx, my)):
                        self._pedir_nombre_participante(permitir_cancelar=True)
                    elif btn_carrera_final.collidepoint((mx, my)):
                        self._pantalla_carrera_todos()
                    elif btn_grabar.collidepoint((mx, my)):
                        return 1
                    elif btn_cola_abrir.collidepoint((mx, my)):
                        with self._bg_lock:
                            hay_cola = bool(self._bg_active_track or self._bg_queue or self._bg_results)
                        if hay_cola:
                            return 3
                    elif btn_cola_cancel.collidepoint((mx, my)):
                        with self._bg_lock:
                            hay_activa = bool(self._bg_active_track or self._bg_queue)
                        if hay_activa:
                            self._cancelar_cola_autonoma()
                            self.estado_menu_msg = "Simulacion autonoma en cola cancelada."
                    elif not self.salud_servidor and btn_copiar_cmd.collidepoint((mx, my)):
                        copiado = False
                        if os.name == "nt":
                            try:
                                subprocess.run(["clip"], input=cmd_backend, text=True, check=True)
                                copiado = True
                            except Exception:
                                pass
                        if not copiado:
                            try:
                                pg.scrap.init()
                                pg.scrap.put(pg.SCRAP_TEXT, cmd_backend.encode("utf-8"))
                                copiado = True
                            except Exception:
                                pass
                        self.estado_menu_msg = (
                            "Comando copiado al portapapeles. Pegalo en otra terminal."
                            if copiado
                            else f"Ejecuta: {cmd_backend}"
                        )
                    else:
                        for i, (est_id, _, _, _, _) in enumerate(tarjetas_estrategia):
                            if card_rects[i].collidepoint((mx, my)):
                                self._pantalla_estrategia(est_id)
                                break

            self.screen.fill(self.BG)
            t1 = self.font_title.render("UIS Grand Prix Foundationals Models", True, self.ACCENT)
            self.screen.blit(t1, (90, 42))

            self.screen.blit(
                self.font_body.render(f"Participante: {self.bridge.student_name}", True, self.TEXTO),
                (90, 86),
            )
            self._dibujar_boton(btn_carrera_final, "[C] Ver Carrera Final (B1-B3)", (52, 36, 16), self.WARN)
            self._dibujar_boton(btn_editar_nombre, "Editar Nombre", self.PANEL_BG, self.BORDE)

            if self.salud_servidor:
                srv_txt = (
                    f"Servidor OK ({self.salud_servidor.get('gpu_nombre', 'CPU')}, "
                    f"VRAM libre: {self.salud_servidor.get('vram_libre_mb', 0)} MB)"
                )
                self.screen.blit(self.font_body.render(srv_txt, True, self.OK), (90, 122))
            else:
                self.screen.blit(
                    self.font_body.render(
                        f"Servidor sin respuesta en {self.bridge.server_url} (Pulsa H para reintentar)",
                        True,
                        self.BAD,
                    ),
                    (90, 120),
                )
                self.screen.blit(
                    self.font_small.render(
                        "Comando para iniciar backend: cd backend_server && python -m uvicorn server:app --host 0.0.0.0 --port 8000",
                        True,
                        self.WARN,
                    ),
                    (90, 148),
                )
                self._dibujar_boton(btn_copiar_cmd, "Copiar comando", (52, 38, 16), self.WARN)

            mouse_p = pg.mouse.get_pos()
            self.screen.blit(
                self.font_h2.render("1. Crear / Editar Dataset:", True, self.TEXTO),
                (90, 174),
            )
            hover_g = btn_grabar.collidepoint(mouse_p)
            pg.draw.rect(self.screen, (24, 36, 58) if hover_g else self.PANEL_BG, btn_grabar, border_radius=12)
            pg.draw.rect(self.screen, self.ACCENT if hover_g else self.BORDE, btn_grabar, 2, border_radius=12)
            self.screen.blit(self.font_h2.render("[G] Grabar Jugadas y Ver Frames (Pistas P1, P2, P3)", True, self.TEXTO), (btn_grabar.x + 24, btn_grabar.y + 16))
            self.screen.blit(self.font_body.render("Conduce con teclado para crear/limpiar data/mis_jugadas.json", True, self.MUTED), (btn_grabar.x + 24, btn_grabar.y + 48))

            self.screen.blit(
                self.font_h2.render("2. Elige una estrategia de adaptacion:", True, self.TEXTO),
                (90, 316),
            )

            for i, (est_id, titulo, desc, bg_col, brd_col) in enumerate(tarjetas_estrategia):
                r = card_rects[i]
                hover = r.collidepoint(mouse_p)
                fill = tuple(min(255, c + 16) for c in bg_col) if hover else bg_col
                pg.draw.rect(self.screen, fill, r, border_radius=14)
                pg.draw.rect(self.screen, brd_col, r, 2, border_radius=14)
                self.screen.blit(self.font_title.render(titulo, True, self.TEXTO), (r.x + 22, r.y + 22))
                # Wrap corto en 2 líneas
                palabras = desc.split()
                l1, l2 = "", ""
                for pal in palabras:
                    if len(l1) + len(pal) + 1 <= 34:
                        l1 = (l1 + " " + pal).strip()
                    else:
                        l2 = (l2 + " " + pal).strip()
                self.screen.blit(self.font_small.render(l1, True, self.TEXTO), (r.x + 22, r.y + 74))
                self.screen.blit(self.font_small.render(l2, True, self.MUTED), (r.x + 22, r.y + 98))
                self.screen.blit(self.font_small.render(f"Entrar ({i + 1}) ->", True, brd_col), (r.x + 22, r.y + 130))

            with self._bg_lock:
                bg_act = self._bg_active_track
                bg_pct = self._bg_active_pct
                bg_q = list(self._bg_queue)
                bg_done = sorted(self._bg_results.keys())

            if bg_act or bg_q:
                cola_txt = f"Cola [3]: Simulando {bg_act} ({bg_pct:.0f}%)" + (f" | Siguientes: {','.join(bg_q)}" if bg_q else "")
                self.screen.blit(self.font_small.render(cola_txt, True, self.WARN), (90, 632))
                self._dibujar_boton(btn_cola_abrir, "Ver Cola [3]", (24, 38, 62), self.ACCENT)
                self._dibujar_boton(btn_cola_cancel, "Cancelar Cola", (56, 20, 26), self.BAD)
            elif bg_done:
                self.screen.blit(
                    self.font_small.render(f"Replays listos en [3]: {', '.join(bg_done)}", True, self.OK),
                    (90, 632),
                )
                self._dibujar_boton(btn_cola_abrir, "Ver Replays [3]", (15, 52, 48), self.OK)

            msg_c = self.WARN if "Error" in self.estado_menu_msg or "Solo" in self.estado_menu_msg else self.TEXTO
            self.screen.blit(self.font_body.render(f"Estado: {self.estado_menu_msg}", True, msg_c), (90, 666))
            self.screen.blit(
                self.font_small.render("Atajos: Teclas 1-4 para elegir modo | H = verificar servidor | ESC = salir", True, self.MUTED),
                (90, 700),
            )
            pg.display.flip()
            self.clock.tick(30)

    def _dibujar_boton(
        self,
        rect: Any,
        texto: str,
        bg: Tuple[int, int, int],
        border: Tuple[int, int, int],
        text_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        pg = self.pygame
        hover = rect.collidepoint(pg.mouse.get_pos())
        fill = tuple(min(255, c + 18) for c in bg) if hover else bg
        pg.draw.rect(self.screen, fill, rect, border_radius=8)
        pg.draw.rect(self.screen, border, rect, 2, border_radius=8)
        surf = self.font_small.render(texto, True, text_color or self.TEXTO)
        tx = rect.x + (rect.w - surf.get_width()) // 2
        ty = rect.y + (rect.h - surf.get_height()) // 2
        self.screen.blit(surf, (tx, ty))

    def _mostrar_controles(self) -> None:
        """Muestra una ventana modal explicando los controles y el contexto del FM."""
        pg = self.pygame
        modal = pg.Rect(210, 150, 800, 440)
        btn_cerrar = pg.Rect(525, 530, 170, 42)
        items = [
            ("Controles de Conduccion y Grabacion", self.ACCENT, True),
            ("  Flechas o W / A / S / D : Conducir", self.TEXTO, False),
            ("  Tecla R / Boton R       : Reinicia el carro al inicio (esquina sup. izq.)", self.TEXTO, False),
            ("  Tecla ESC               : Guardar jugadas y volver al menu", self.TEXTO, False),
            ("", self.TEXTO, False),
            ("Botones del Panel (Etiquetador)", self.ACCENT, True),
            ("  P1 | P2 | P3            : Selecciona entre las 3 pistas de entrenamiento", self.MUTED, False),
            ("  - / + (Cada N frames)   : Frecuencia de captura de muestras", self.MUTED, False),
            ("  Iniciar / Detener       : Activa o pausa la grabacion de frames", self.MUTED, False),
            ("  Ver Frames              : Revisa frames (A/D o <-/->) y borra por pista", self.MUTED, False),
            ("  Guardar y Salir         : Guarda mis_jugadas.json y vuelve al menu", self.MUTED, False),
            ("", self.TEXTO, False),
            ("Variables de Entrada y Salida", self.WARN, True),
            ("  Entrada : 5 rayos, cp_dist, velocidad e Historial (ultimas 5)", self.MUTED, False),
            ("  Salida  : accion (0=recto, 1=izq, 2=der, 3=frenar)", self.MUTED, False),
        ]

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN and event.key in (pg.K_ESCAPE, pg.K_RETURN, pg.K_KP_ENTER):
                    return
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_cerrar.collidepoint(event.pos):
                        return

            pg.draw.rect(self.screen, (10, 16, 28), modal, border_radius=14)
            pg.draw.rect(self.screen, self.ACCENT, modal, 2, border_radius=14)
            y = 175
            for txt, col, is_hdr in items:
                if txt:
                    font = self.font_h2 if is_hdr else self.font_small
                    self.screen.blit(font.render(txt, True, col), (240, y))
                y += 24
            self._dibujar_boton(btn_cerrar, "Entendido", (15, 52, 48), self.ACCENT)
            pg.display.flip()
            self.clock.tick(30)

    def _confirmar_texto(self, titulo: str, mensaje: str, palabra_clave: str) -> bool:
        """Diálogo de confirmación que exige escribir exactamente `palabra_clave`."""
        pg = self.pygame
        texto_ingresado = ""
        modal = pg.Rect(240, 230, 740, 270)
        caja_txt = pg.Rect(280, 365, 660, 42)
        btn_ok = pg.Rect(580, 430, 170, 42)
        btn_cancel = pg.Rect(770, 430, 170, 42)

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return False
                    if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        if texto_ingresado.strip() == palabra_clave:
                            return True
                    elif event.key == pg.K_BACKSPACE:
                        texto_ingresado = texto_ingresado[:-1]
                    elif event.unicode and event.unicode.isprintable() and len(texto_ingresado) < 25:
                        texto_ingresado += event.unicode
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if btn_cancel.collidepoint(pos):
                        return False
                    if btn_ok.collidepoint(pos) and texto_ingresado.strip() == palabra_clave:
                        return True

            pg.draw.rect(self.screen, (10, 16, 28), modal, border_radius=14)
            pg.draw.rect(self.screen, self.BAD, modal, 2, border_radius=14)
            self.screen.blit(self.font_h2.render(titulo, True, self.BAD), (280, 255))
            self.screen.blit(self.font_body.render(mensaje, True, self.TEXTO), (280, 295))
            self.screen.blit(
                self.font_body.render(f"Escribe  {palabra_clave}  para confirmar:", True, self.WARN),
                (280, 328),
            )
            pg.draw.rect(self.screen, (20, 29, 46), caja_txt, border_radius=8)
            pg.draw.rect(self.screen, self.ACCENT, caja_txt, 2, border_radius=8)
            self.screen.blit(self.font_h2.render(texto_ingresado + "|", True, self.TEXTO), (caja_txt.x + 12, caja_txt.y + 10))

            listo = texto_ingresado.strip() == palabra_clave
            self._dibujar_boton(btn_ok, "Confirmar", (75, 22, 34) if listo else (30, 35, 45), self.BAD if listo else self.BORDE)
            self._dibujar_boton(btn_cancel, "Cancelar", self.PANEL_BG, self.BORDE)
            pg.display.flip()
            self.clock.tick(30)

    def _dibujar_miniatura_frame(
        self,
        rect: Any,
        muestra: Dict[str, Any],
        num_frame: int,
        seleccionada: bool,
    ) -> None:
        """Dibuja una miniatura de la pista, carro y rayos para la vista en cuadrícula."""
        pg = self.pygame
        bg = (22, 34, 56) if seleccionada else self.PANEL_BG
        border = self.ACCENT if seleccionada else self.BORDE
        pg.draw.rect(self.screen, bg, rect, border_radius=10)
        pg.draw.rect(self.screen, border, rect, 2 if seleccionada else 1, border_radius=10)

        pista_k = str(muestra.get("pista", "P1"))
        if pista_k not in PISTAS:
            pista_k = "P1"
        pista = PISTAS[pista_k]
        escala = min((rect.w - 22) / 22.0, (rect.h - 42) / 15.0)
        cx_box = rect.centerx
        cy_box = rect.y + (rect.h - 22) // 2

        def w2m(wx: float, wy: float) -> Tuple[int, int]:
            return int(cx_box + wx * escala), int(cy_box + wy * escala)

        pts = [w2m(px, py) for px, py in pista["puntos"]]
        ancho_px = max(4, int(pista["ancho_medio"] * 2.0 * escala))
        pg.draw.lines(self.screen, self.BORDE_PISTA, True, pts, ancho_px + 3)
        pg.draw.lines(self.screen, self.ASFALTO, True, pts, ancho_px)
        for sx, sy in pts:
            pg.draw.circle(self.screen, self.ASFALTO, (sx, sy), max(2, ancho_px // 2))

        est_info = muestra.get("estado")
        sens = muestra.get("sensores", [0.0] * 5)
        if isinstance(est_info, dict) and "x" in est_info and "y" in est_info:
            est_vis = EstadoCarro(
                x=float(est_info["x"]),
                y=float(est_info["y"]),
                angulo=float(est_info.get("angulo", 0.0)),
                velocidad=float(est_info.get("velocidad", 0.0)),
                tiempo=float(est_info.get("tiempo", 0.0)),
                siguiente_checkpoint=int(est_info.get("siguiente_checkpoint", 1)) % len(pista["puntos"]),
            )
            _, impactos = calcular_sensores_rayos(est_vis, pista_k)
            cp_pt = pista["puntos"][est_vis.siguiente_checkpoint]
            car_s = w2m(est_vis.x, est_vis.y)
            pg.draw.line(self.screen, (56, 189, 248), car_s, w2m(cp_pt[0], cp_pt[1]), 1)
            for val, (hx, hy) in zip(sens, impactos):
                c_r = self.OK if val > 0.5 else (self.WARN if val > 0.25 else self.BAD)
                pg.draw.line(self.screen, c_r, car_s, w2m(hx, hy), 1)
            pg.draw.circle(self.screen, self.CAR_COLOR, car_s, max(3, int(0.45 * escala)))

        accion = int(muestra.get("accion", 0))
        nom_corto = ["Recto", "Izq", "Der", "Frenar"][accion % 4]
        lbl = self.font_small.render(f"#{num_frame} | {pista_k} | {accion}:{nom_corto}", True, self.TEXTO if seleccionada else self.MUTED)
        self.screen.blit(lbl, (rect.x + 10, rect.bottom - 22))

    def _pantalla_inspector_frames(self, jugadas: List[Dict[str, Any]], pista_inicial: str = "TODAS") -> List[Dict[str, Any]]:
        pg = self.pygame
        filtro = pista_inicial if pista_inicial in PISTAS_TRAIN else "TODAS"
        idx = 0
        vista_cuadricula = False
        pila_deshacer: List[List[Dict[str, Any]]] = []
        x0 = self.AREA_PISTA_W + 18
        bw = self.ANCHO - self.AREA_PISTA_W - 36
        half_w = (bw - 10) // 2
        tab_w = (bw - 18) // 4

        def guardar_estado_undo() -> None:
            pila_deshacer.append(list(jugadas))
            if len(pila_deshacer) > 50:
                pila_deshacer.pop(0)

        while True:
            indices_vis = [
                i for i, j in enumerate(jugadas)
                if filtro == "TODAS" or j.get("pista") == filtro
            ]
            if indices_vis:
                idx = max(0, min(idx, len(indices_vis) - 1))
            else:
                idx = 0

            # Pestañas de filtro agrupadas por pista
            tabs = ["TODAS", "P1", "P2", "P3"]
            tab_rects = [pg.Rect(x0 + t_i * (tab_w + 6), 430, tab_w, 30) for t_i in range(4)]

            btn_prev = pg.Rect(x0, 468, half_w, 36)
            btn_next = pg.Rect(x0 + half_w + 10, 468, half_w, 36)
            btn_grid = pg.Rect(x0, 512, bw, 36)
            btn_del_one = pg.Rect(x0, 556, half_w, 36)
            btn_del_pista = pg.Rect(x0 + half_w + 10, 556, half_w, 36)
            btn_clear_all = pg.Rect(x0, 600, half_w, 36)
            btn_undo = pg.Rect(x0 + half_w + 10, 600, half_w, 36)
            btn_back = pg.Rect(x0, 646, bw, 40)

            # Rectángulos 3x3 para miniaturas en modo cuadrícula
            grid_rects: List[Any] = []
            if vista_cuadricula:
                gw = (self.AREA_PISTA_W - 48) // 3
                gh = (self.ALTO - 48) // 3
                for r_i in range(3):
                    for c_i in range(3):
                        grid_rects.append(pg.Rect(14 + c_i * (gw + 10), 14 + r_i * (gh + 10), gw, gh))

            pista_obj = filtro
            if indices_vis:
                pista_obj = str(jugadas[indices_vis[idx]].get("pista", filtro))

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return jugadas
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return jugadas
                    if (event.mod & pg.KMOD_CTRL) and event.key == pg.K_z and pila_deshacer:
                        jugadas = pila_deshacer.pop()
                        guardar_jugadas(jugadas)
                    elif event.key in (pg.K_LEFT, pg.K_a) and indices_vis:
                        idx = (idx - 1) % len(indices_vis)
                    elif event.key in (pg.K_RIGHT, pg.K_d) and indices_vis:
                        idx = (idx + 1) % len(indices_vis)
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if btn_back.collidepoint(pos):
                        return jugadas
                    for t_name, t_rect in zip(tabs, tab_rects):
                        if t_rect.collidepoint(pos):
                            filtro = t_name
                            idx = 0
                    if btn_grid.collidepoint(pos):
                        vista_cuadricula = not vista_cuadricula
                    if vista_cuadricula and indices_vis:
                        page_start = (idx // 9) * 9
                        for cell_i, g_rect in enumerate(grid_rects):
                            cand = page_start + cell_i
                            if cand < len(indices_vis) and g_rect.collidepoint(pos):
                                if idx == cand:
                                    vista_cuadricula = False
                                else:
                                    idx = cand
                                break
                    if btn_prev.collidepoint(pos) and indices_vis:
                        paso = 9 if vista_cuadricula else 1
                        idx = (idx - paso) % len(indices_vis)
                    if btn_next.collidepoint(pos) and indices_vis:
                        paso = 9 if vista_cuadricula else 1
                        idx = (idx + paso) % len(indices_vis)
                    if btn_del_one.collidepoint(pos) and indices_vis:
                        guardar_estado_undo()
                        jugadas.pop(indices_vis[idx])
                        guardar_jugadas(jugadas)
                    if btn_del_pista.collidepoint(pos) and indices_vis and pista_obj in PISTAS_TRAIN:
                        n_pista = sum(1 for j in jugadas if j.get("pista") == pista_obj)
                        clave_conf = f"borrar-{pista_obj.lower()}"
                        if self._confirmar_texto(
                            f"ATENCION: Borrar frames de {pista_obj}",
                            f"Se eliminaran los {n_pista} frames grabados de {pista_obj}.",
                            clave_conf,
                        ):
                            guardar_estado_undo()
                            jugadas = [j for j in jugadas if j.get("pista") != pista_obj]
                            guardar_jugadas(jugadas)
                            idx = 0
                    if btn_clear_all.collidepoint(pos) and jugadas:
                        if self._confirmar_texto(
                            "ATENCION: Se borraran TODAS las grabaciones",
                            f"Se eliminaran los {len(jugadas)} frames de todas las pistas (P1, P2 y P3).",
                            "borrar-todo",
                        ):
                            guardar_estado_undo()
                            jugadas.clear()
                            guardar_jugadas(jugadas)
                            idx = 0
                    if btn_undo.collidepoint(pos) and pila_deshacer:
                        jugadas = pila_deshacer.pop()
                        guardar_jugadas(jugadas)

            # Recalcular indices visibles tras posibles eliminaciones en este frame
            indices_vis = [
                i for i, j in enumerate(jugadas)
                if filtro == "TODAS" or j.get("pista") == filtro
            ]
            if indices_vis:
                idx = max(0, min(idx, len(indices_vis) - 1))
            else:
                idx = 0

            conteo = conteo_por_pista(jugadas)
            if not indices_vis:
                self.screen.fill(self.BG)
                self.screen.blit(
                    self.font_title.render(f"Inspector de Frames ({filtro})", True, self.ACCENT),
                    (90, 60),
                )
                self.screen.blit(
                    self.font_h2.render(f"No hay frames grabados en {filtro}.", True, self.WARN),
                    (90, 120),
                )
                lineas = [
                    ("INSPECTOR POR PISTA", self.ACCENT),
                    (f"Filtro: {filtro} (0 frames)", self.WARN),
                    (f"P1:{conteo['P1']} | P2:{conteo['P2']} | P3:{conteo['P3']}", self.MUTED),
                ]
                self._dibujar_panel_lateral(lineas, [0.0] * 5, velocidad_norm=0.0)
            else:
                m = jugadas[indices_vis[idx]]
                accion = int(m.get("accion", 0))
                nom_acc = ACCIONES_DISCRETAS[accion % len(ACCIONES_DISCRETAS)][0]
                pista_k = m.get("pista", "P1")
                if pista_k not in PISTAS:
                    pista_k = "P1"
                hist = m.get("historial", [])
                sens = m.get("sensores", [0.0] * 5)
                est_info = m.get("estado")
                vel_norm = m.get("velocidad")
                if vel_norm is None and isinstance(est_info, dict) and "velocidad" in est_info:
                    vel_norm = velocidad_normalizada(float(est_info["velocidad"]))

                if vista_cuadricula:
                    pg.draw.rect(self.screen, self.BG, (0, 0, self.AREA_PISTA_W, self.ALTO))
                    page_start = (idx // 9) * 9
                    for cell_i, g_rect in enumerate(grid_rects):
                        cand = page_start + cell_i
                        if cand >= len(indices_vis):
                            break
                        self._dibujar_miniatura_frame(
                            g_rect,
                            jugadas[indices_vis[cand]],
                            cand + 1,
                            cand == idx,
                        )
                elif isinstance(est_info, dict) and "x" in est_info and "y" in est_info:
                    est_vis = EstadoCarro(
                        x=float(est_info["x"]),
                        y=float(est_info["y"]),
                        angulo=float(est_info.get("angulo", 0.0)),
                        velocidad=float(est_info.get("velocidad", 0.0)),
                        tiempo=float(est_info.get("tiempo", 0.0)),
                        siguiente_checkpoint=int(est_info.get("siguiente_checkpoint", 1)) % len(PISTAS[pista_k]["puntos"]),
                        checkpoints=int(est_info.get("checkpoints", 0)),
                        vueltas=int(est_info.get("vueltas", 0)),
                    )
                    _, impactos = calcular_sensores_rayos(est_vis, pista_k)
                    self._dibujar_pista_y_carro(pista_k, est_vis, sens, impactos)
                else:
                    est_vis = estado_inicial(pista_k)
                    self._dibujar_pista_y_carro(pista_k, est_vis, [0.0] * 5, [(est_vis.x, est_vis.y)] * 5)

                vel_txt = f"{est_info['velocidad']:.2f} m/s" if isinstance(est_info, dict) and "velocidad" in est_info else "N/A"
                cp_txt = f"CP {est_info.get('siguiente_checkpoint', '?')}" if isinstance(est_info, dict) else "CP ?"
                lineas = [
                    ("INSPECTOR POR PISTA", self.ACCENT),
                    (f"Frame #{idx + 1}/{len(indices_vis)} ({filtro}) | {cp_txt}", self.OK),
                    (f"P1:{conteo['P1']} | P2:{conteo['P2']} | P3:{conteo['P3']}", self.MUTED),
                    (f"Vel: {vel_txt} | <- / -> o A / D", self.TEXTO),
                ]
                cp_d = m.get("cp_dist")
                cp_a = m.get("cp_ang")
                cp_val: Optional[Union[float, Tuple[float, float]]] = (
                    (float(cp_d), float(cp_a))
                    if (cp_d is not None and cp_a is not None)
                    else (float(cp_d) if cp_d is not None else None)
                )
                self._dibujar_panel_lateral(
                    lineas,
                    sens,
                    velocidad_norm=vel_norm if vel_norm is not None else 0.0,
                    historial=hist,
                    accion_texto=f"{accion} ({nom_acc})",
                    cp_dist=cp_val,
                )

            for t_name, t_rect in zip(tabs, tab_rects):
                activo = t_name == filtro
                self._dibujar_boton(
                    t_rect,
                    t_name,
                    (15, 52, 48) if activo else self.PANEL_BG,
                    self.ACCENT if activo else self.BORDE,
                )

            self._dibujar_boton(btn_prev, "< Anterior", self.PANEL_BG, self.BORDE)
            self._dibujar_boton(btn_next, "Siguiente >", self.PANEL_BG, self.BORDE)
            lbl_grid = "Vista Individual" if vista_cuadricula else "Visualizar Cuadricula"
            self._dibujar_boton(btn_grid, lbl_grid, (24, 38, 62), self.ACCENT)
            self._dibujar_boton(btn_del_one, "Eliminar Frame", (55, 22, 32), self.BAD)
            lbl_pista = f"Borrar {pista_obj}" if pista_obj in PISTAS_TRAIN else "Borrar Pista"
            self._dibujar_boton(btn_del_pista, lbl_pista, (60, 28, 20), self.WARN)
            self._dibujar_boton(btn_clear_all, "Borrar Todo", (70, 18, 28), self.BAD)
            lbl_undo = f"Ctrl+Z ({len(pila_deshacer)})" if pila_deshacer else "Ctrl+Z (0)"
            self._dibujar_boton(
                btn_undo,
                lbl_undo,
                (24, 38, 62) if pila_deshacer else (22, 26, 36),
                self.ACCENT if pila_deshacer else self.BORDE,
            )
            self._dibujar_boton(btn_back, "Volver a Grabacion", (15, 52, 48), self.ACCENT)
            pg.display.flip()
            self.clock.tick(30)

    def _modo_grabar_jugadas(self) -> None:
        pg = self.pygame
        idx_pista = 0
        clave = PISTAS_TRAIN[idx_pista]
        est = estado_inicial(clave)
        jugadas = cargar_jugadas()
        ventana = max(1, min(8, int(cfg.CONFIG_DATOS.get("ventana_historia", 5))))
        hist: List[int] = [0] * ventana
        frame = 0
        grabando = False
        cada_n_frames = RECORD_EVERY_N_FRAMES
        frames_sesion = 0

        x0 = self.AREA_PISTA_W + 18
        bw = self.ANCHO - self.AREA_PISTA_W - 36
        tab3_w = (bw - 16) // 3
        btn_reiniciar = pg.Rect(16, 16, 40, 40)
        btn_controles = pg.Rect(x0, 385, bw, 36)
        pista_btns = [pg.Rect(x0 + i * (tab3_w + 8), 431, tab3_w, 34) for i in range(3)]
        btn_freq_menos = pg.Rect(x0, 487, 56, 36)
        btn_freq_mas = pg.Rect(x0 + bw - 56, 487, 56, 36)
        btn_rec = pg.Rect(x0, 533, bw, 42)
        btn_ver_frames = pg.Rect(x0, 587, bw, 62)
        btn_guardar_salir = pg.Rect(x0, 661, bw, 42)

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    guardar_jugadas(jugadas)
                    return
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        guardar_jugadas(jugadas)
                        self.estado_menu_msg = f"Jugadas guardadas en data/mis_jugadas.json ({len(jugadas)} muestras)."
                        return
                    if event.key == pg.K_r:
                        est = estado_inicial(clave)
                        hist = [0] * ventana
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if btn_rec.collidepoint(pos):
                        grabando = not grabando
                    elif btn_freq_menos.collidepoint(pos):
                        cada_n_frames = max(1, cada_n_frames - 1)
                    elif btn_freq_mas.collidepoint(pos):
                        cada_n_frames = min(30, cada_n_frames + 1)
                    elif any(r.collidepoint(pos) for r in pista_btns):
                        for i, r in enumerate(pista_btns):
                            if r.collidepoint(pos):
                                idx_pista = i
                                clave = PISTAS_TRAIN[idx_pista]
                                est = estado_inicial(clave)
                                hist = [0] * ventana
                                break
                    elif btn_reiniciar.collidepoint(pos):
                        est = estado_inicial(clave)
                        hist = [0] * ventana
                    elif btn_controles.collidepoint(pos):
                        grabando = False
                        self._mostrar_controles()
                    elif btn_ver_frames.collidepoint(pos):
                        grabando = False
                        guardar_jugadas(jugadas)
                        jugadas = self._pantalla_inspector_frames(jugadas, pista_inicial=clave)
                    elif btn_guardar_salir.collidepoint(pos):
                        guardar_jugadas(jugadas)
                        self.estado_menu_msg = f"Grabacion guardada ({len(jugadas)} muestras totales, +{frames_sesion} nuevas)."
                        return

            sensores, impactos = calcular_sensores_rayos(est, clave)
            keys = pg.key.get_pressed()
            conduciendo = False

            if keys[pg.K_DOWN] or keys[pg.K_s]:
                accion = 3
                conduciendo = True
            elif keys[pg.K_LEFT] or keys[pg.K_a]:
                accion = 1
                conduciendo = True
            elif keys[pg.K_RIGHT] or keys[pg.K_d]:
                accion = 2
                conduciendo = True
            elif keys[pg.K_UP] or keys[pg.K_w]:
                accion = 0
                conduciendo = True
            else:
                accion = 3

            if grabando and (conduciendo or est.velocidad > 0.05) and frame % cada_n_frames == 0:
                cp_dist, cp_ang = info_checkpoint(est, clave)
                jugadas.append({
                    "pista": clave,
                    "sensores": sensores,
                    "velocidad": velocidad_normalizada(est.velocidad),
                    "cp_dist": cp_dist,
                    "cp_ang": cp_ang,
                    "historial": hist[-ventana:],
                    "accion": accion,
                    "estado": {
                        "x": round(est.x, 3),
                        "y": round(est.y, 3),
                        "angulo": round(est.angulo, 4),
                        "velocidad": round(est.velocidad, 2),
                        "tiempo": round(est.tiempo, 2),
                        "siguiente_checkpoint": int(est.siguiente_checkpoint),
                        "checkpoints": int(est.checkpoints),
                        "vueltas": int(est.vueltas),
                    },
                })
                frames_sesion += 1

            if conduciendo or est.velocidad > 0.02:
                paso_simulacion(est, accion, clave)
                hist.append(accion)
            frame += 1

            self._dibujar_pista_y_carro(clave, est, sensores, impactos)
            self._dibujar_boton(btn_reiniciar, "R", self.PANEL_BG, self.BORDE)
            conteo = conteo_por_pista(jugadas)
            accion_txt = ACCIONES_DISCRETAS[accion][0]
            lineas = [
                ("MODO [1]: GRABAR JUGADAS", self.ACCENT),
                (f"Pista: {PISTAS[clave]['nombre']}", self.TEXTO),
            ]
            self._dibujar_panel_lateral(
                lineas,
                sensores,
                velocidad_norm=velocidad_normalizada(est.velocidad),
                historial=hist[-ventana:],
                accion_texto=accion_txt,
                cp_dist=info_checkpoint(est, clave),
            )

            # Botones interactivos en el panel derecho
            self._dibujar_boton(btn_controles, "Instrucciones (Controles)", (24, 38, 62), self.ACCENT)
            for p_name, p_rect in zip(PISTAS_TRAIN, pista_btns):
                activa = p_name == clave
                self._dibujar_boton(
                    p_rect,
                    p_name,
                    (15, 52, 48) if activa else self.PANEL_BG,
                    self.ACCENT if activa else self.BORDE,
                )

            self._dibujar_boton(btn_freq_menos, "-", self.PANEL_BG, self.BORDE)
            freq_txt = self.font_small.render(f"Cada {cada_n_frames} frames", True, self.TEXTO)
            self.screen.blit(freq_txt, (x0 + (bw - freq_txt.get_width()) // 2, 496))
            self._dibujar_boton(btn_freq_mas, "+", self.PANEL_BG, self.BORDE)

            if grabando:
                self._dibujar_boton(btn_rec, f"(+{frames_sesion}) DETENER GRABACION", (75, 22, 34), self.BAD)
            else:
                self._dibujar_boton(btn_rec, f"(+{frames_sesion}) INICIAR GRABACION", (16, 66, 42), self.OK)

            excede_300 = len(jugadas) > 300
            self._dibujar_boton(
                btn_ver_frames,
                "",
                (68, 22, 28) if excede_300 else (24, 38, 62),
                self.BAD if excede_300 else self.ACCENT,
            )
            l1 = self.font_body.render(
                f"Ver Frames: {len(jugadas)} / 300" if excede_300 else f"Ver Frames: {len(jugadas)}",
                True,
                self.BAD if excede_300 else self.TEXTO,
            )
            l2 = self.font_small.render(
                f"P1: {conteo['P1']}  |  P2: {conteo['P2']}  |  P3: {conteo['P3']}",
                True,
                self.BAD if excede_300 else self.ACCENT,
            )
            self.screen.blit(l1, (btn_ver_frames.x + (btn_ver_frames.w - l1.get_width()) // 2, btn_ver_frames.y + 10))
            self.screen.blit(l2, (btn_ver_frames.x + (btn_ver_frames.w - l2.get_width()) // 2, btn_ver_frames.y + 35))
            self._dibujar_boton(btn_guardar_salir, "Guardar y Salir", (15, 52, 48), self.ACCENT)

            pg.display.flip()
            self.clock.tick(30)

    def _elegir_lora_a_sobrescribir(self, existentes: List[str]) -> Optional[str]:
        pg = self.pygame
        btns = [pg.Rect(240, 270 + i * 64, self.ANCHO - 480, 48) for i in range(len(existentes))]
        btn_cancel = pg.Rect(self.ANCHO // 2 - 110, 280 + len(existentes) * 64, 220, 44)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    return None
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if btn_cancel.collidepoint(pos):
                        return None
                    for i, b in enumerate(btns):
                        if b.collidepoint(pos):
                            return existentes[i]

            self.screen.fill(self.BG)
            self.screen.blit(
                self.font_title.render("Limite de 3 modelos LoRA alcanzado", True, self.WARN),
                (240, 170),
            )
            self.screen.blit(
                self.font_body.render("Elige cual de tus 3 adaptadores deseas sobrescribir con este nuevo entrenamiento:", True, self.TEXTO),
                (240, 215),
            )
            for i, b in enumerate(btns):
                self._dibujar_boton(b, f"Sobrescribir: {existentes[i]}", (56, 38, 16), self.WARN)
            self._dibujar_boton(btn_cancel, "Cancelar", self.PANEL_BG, self.BORDE)
            pg.display.flip()
            self.clock.tick(30)

    def _modo_entrenar_lora(self) -> None:
        pg = self.pygame
        if cfg.ESTRATEGIA.upper().strip() not in {"LORA", "LORA_ICL"}:
            self.estado_menu_msg = "Solo para estrategias LORA o LORA+ICL. Cambia la estrategia arriba."
            return
        jugadas = cargar_jugadas()
        if not jugadas:
            self.estado_menu_msg = "Error: no hay jugadas en data/mis_jugadas.json. Graba primero en [1]."
            return

        existentes = self.bridge.listar_adaptadores()
        if len(existentes) >= 3:
            a_borrar = self._elegir_lora_a_sobrescribir(existentes[:3])
            if not a_borrar:
                self.estado_menu_msg = "Entrenamiento LoRA cancelado."
                return
            self.bridge.eliminar_adaptador(a_borrar)
            existentes = self.bridge.listar_adaptadores()

        nuevo_nombre = f"v{len(existentes) + 1}_{time.strftime('%H%M%S')}"
        cancel_ev = threading.Event()
        prog_lock = threading.Lock()
        prog_state: Dict[str, Any] = {
            "pct": 0.0,
            "epoch": 0,
            "total_epochs": int(cfg.CONFIG_LORA.get("epochs", 8)),
            "loss": None,
            "done": False,
            "res": None,
            "err": None,
            "cancelled": False,
        }

        def _on_prog(job: Dict[str, Any]) -> None:
            with prog_lock:
                prog_state["pct"] = float(job.get("pct", prog_state["pct"]))
                prog_state["epoch"] = int(job.get("epoch", prog_state["epoch"]))
                prog_state["total_epochs"] = int(job.get("total_epochs", prog_state["total_epochs"]))
                prog_state["loss"] = job.get("loss")

        def _worker_train() -> None:
            try:
                res = self.bridge.entrenar_lora(
                    jugadas,
                    cfg.CONFIG_LORA,
                    cfg.CONFIG_DATOS,
                    adapter_name=nuevo_nombre,
                    on_progress=_on_prog,
                    cancel_event=cancel_ev,
                )
                with prog_lock:
                    prog_state["pct"] = 100.0
                    prog_state["res"] = res
                    prog_state["done"] = True
            except InterruptedError:
                with prog_lock:
                    prog_state["cancelled"] = True
                    prog_state["done"] = True
            except Exception as exc:
                with prog_lock:
                    prog_state["err"] = str(exc)
                    prog_state["done"] = True

        t_train = threading.Thread(target=_worker_train, daemon=True)
        t_train.start()
        t_inicio = time.monotonic()
        barra_bg = pg.Rect(140, 340, self.ANCHO - 280, 34)
        btn_cancelar = pg.Rect(self.ANCHO // 2 - 110, 465, 220, 44)

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    cancel_ev.set()
                    return
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    cancel_ev.set()
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_cancelar.collidepoint(event.pos):
                        cancel_ev.set()

            with prog_lock:
                done = bool(prog_state["done"])
                pct = float(prog_state["pct"])
                ep = int(prog_state["epoch"])
                tot_ep = int(prog_state["total_epochs"])
                loss_v = prog_state["loss"]
                res_ok = prog_state["res"]
                err_msg = prog_state["err"]
                was_cancelled = bool(prog_state["cancelled"])

            if done:
                if was_cancelled or cancel_ev.is_set():
                    self.estado_menu_msg = "Entrenamiento LoRA cancelado."
                elif err_msg:
                    self.estado_menu_msg = f"Error LoRA: {err_msg}"
                elif res_ok:
                    self.loras_disponibles = self.bridge.listar_adaptadores()
                    self.lora_activo = nuevo_nombre
                    self.estado_menu_msg = (
                        f"LoRA '{nuevo_nombre}' OK ({res_ok['ejemplos_entrenados']} ej, "
                        f"loss={res_ok['loss_final']}, t={res_ok['tiempo_entrenamiento_s']}s)."
                    )
                return

            self.screen.fill(self.BG)
            self.screen.blit(
                self.font_title.render(f"Entrenando adaptador LoRA '{nuevo_nombre}'...", True, self.ACCENT),
                (140, 175),
            )
            self.screen.blit(
                self.font_body.render(
                    "Puedes cancelar en cualquier momento; el servidor detendra el entrenamiento entre batches.",
                    True,
                    self.MUTED,
                ),
                (140, 225),
            )

            pg.draw.rect(self.screen, self.PANEL_BG, barra_bg, border_radius=8)
            pg.draw.rect(self.screen, self.BORDE, barra_bg, 2, border_radius=8)
            fill_w = max(0, int((barra_bg.w - 4) * (pct / 100.0)))
            if fill_w > 0:
                pg.draw.rect(
                    self.screen,
                    (129, 140, 248),
                    pg.Rect(barra_bg.x + 2, barra_bg.y + 2, fill_w, barra_bg.h - 4),
                    border_radius=6,
                )
            pct_txt = self.font_h2.render(f"{pct:.0f}%", True, self.TEXTO)
            self.screen.blit(pct_txt, (barra_bg.centerx - pct_txt.get_width() // 2, barra_bg.y + 4))

            loss_str = f"{loss_v:.4f}" if isinstance(loss_v, (int, float)) else "calculando..."
            trans = time.monotonic() - t_inicio
            self.screen.blit(
                self.font_body.render(
                    f"Epoca: {ep} / {tot_ep} | Loss: {loss_str} | Tiempo: {trans:.1f}s",
                    True,
                    self.TEXTO,
                ),
                (140, 398),
            )
            self._dibujar_boton(
                btn_cancelar,
                "Cancelando..." if cancel_ev.is_set() else "Cancelar Entrenamiento (ESC)",
                (56, 20, 26),
                self.BAD,
            )
            pg.display.flip()
            self.clock.tick(30)

    def _correr_pista_autonoma_visual(
        self, clave_pista: str, jugadas_ref: List[Dict[str, Any]], titulo_modo: str
    ) -> Optional[Dict[str, Any]]:
        pg = self.pygame
        est = estado_inicial(clave_pista)
        ventana = max(1, min(8, int(cfg.CONFIG_DATOS.get("ventana_historia", 5))))
        hist: List[int] = [0] * ventana
        latencias_rtt_ms: List[float] = []
        frames_grabados: List[Dict[str, Any]] = []

        cancel_event = threading.Event()
        lock = threading.Lock()
        estado_carga: Dict[str, Any] = {
            "pct": 0.0,
            "frames": 0,
            "tiempo": 0.0,
            "progreso_pista": 0.0,
            "rtt_ms": 0.0,
            "gpu_ms": 0.0,
            "error": "",
            "done": False,
            "exc": None,
        }

        def _simular_en_segundo_plano() -> None:
            accion_actual = 0
            frame = 0
            ultimo_gpu_ms = 0.0
            ultimo_rtt_ms = 0.0
            ultimo_err = ""
            try:
                while (
                    not cancel_event.is_set()
                    and est.tiempo_carrera < TIEMPO_LIMITE_PRUEBA
                    and est.vueltas < VUELTAS_OBJETIVO
                ):
                    sensores, impactos = calcular_sensores_rayos(est, clave_pista)
                    cp_dist_ang = info_checkpoint(est, clave_pista)

                    if frame % DECIDE_EVERY_N_FRAMES == 0:
                        resp = self.bridge.decidir_accion(
                            estrategia=cfg.ESTRATEGIA,
                            estado_actual=sensores,
                            historial=hist[-ventana:],
                            jugadas_referencia=jugadas_ref,
                            config_icl=cfg.CONFIG_ICL,
                            config_datos=cfg.CONFIG_DATOS,
                            velocidad=velocidad_normalizada(est.velocidad),
                            cp_dist=cp_dist_ang[0],
                            cp_ang=cp_dist_ang[1],
                            adapter_name=self.lora_activo,
                        )
                        accion_actual = int(resp["accion"])
                        ultimo_rtt_ms = float(resp["rtt_ms"])
                        ultimo_gpu_ms = float(resp["latencia_gpu_ms"])
                        ultimo_err = str(resp.get("error") or "")
                        latencias_rtt_ms.append(ultimo_rtt_ms)

                    lat_acum_s = sum(latencias_rtt_ms) / 1000.0
                    frames_grabados.append({
                        "est": copy.deepcopy(est),
                        "sensores": list(sensores),
                        "impactos": list(impactos),
                        "accion": accion_actual,
                        "hist": list(hist[-ventana:]),
                        "cp_dist_ang": cp_dist_ang,
                        "lat_acum_s": lat_acum_s,
                        "rtt_ms": ultimo_rtt_ms,
                        "gpu_ms": ultimo_gpu_ms,
                        "err": ultimo_err,
                        "progreso_pct": calcular_progreso_pct(est, clave_pista),
                    })

                    paso_simulacion(est, accion_actual, clave_pista)
                    hist.append(accion_actual)
                    frame += 1

                    prog_pista = calcular_progreso_pct(est, clave_pista)
                    pct_tiempo = (est.tiempo_carrera / TIEMPO_LIMITE_PRUEBA) * 100.0
                    pct_total = min(99.0, max(prog_pista if est.vueltas >= VUELTAS_OBJETIVO else 0.0, pct_tiempo))
                    if est.vueltas >= VUELTAS_OBJETIVO:
                        pct_total = 100.0
                    with lock:
                        estado_carga["pct"] = pct_total
                        estado_carga["frames"] = frame
                        estado_carga["tiempo"] = est.tiempo_carrera
                        estado_carga["progreso_pista"] = prog_pista
                        estado_carga["rtt_ms"] = ultimo_rtt_ms
                        estado_carga["gpu_ms"] = ultimo_gpu_ms
                        estado_carga["error"] = ultimo_err

                with lock:
                    estado_carga["pct"] = 100.0
                    estado_carga["done"] = True
            except Exception as exc:
                with lock:
                    estado_carga["exc"] = str(exc)
                    estado_carga["done"] = True

        hilo = threading.Thread(target=_simular_en_segundo_plano, daemon=True)
        hilo.start()

        # Fase 1: Pantalla de carga mientras la GPU procesa toda la pista en segundo plano
        barra_bg = pg.Rect(140, 350, self.ANCHO - 280, 34)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    cancel_event.set()
                    return None
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    cancel_event.set()
                    return None

            with lock:
                done = bool(estado_carga["done"])
                pct = float(estado_carga["pct"])
                n_frames = int(estado_carga["frames"])
                t_fis = float(estado_carga["tiempo"])
                p_pista = float(estado_carga["progreso_pista"])
                rtt = float(estado_carga["rtt_ms"])
                gpu = float(estado_carga["gpu_ms"])
                err = str(estado_carga["error"])
                exc_msg = estado_carga["exc"]

            if exc_msg:
                self.estado_menu_msg = f"Error en simulacion: {exc_msg}"
                return None
            if done:
                break

            self.screen.fill(self.BG)
            self.screen.blit(self.font_title.render(titulo_modo, True, self.ACCENT), (140, 140))
            self.screen.blit(
                self.font_h2.render(
                    f"Calculando simulacion en servidor ({PISTAS[clave_pista]['nombre']} | {cfg.ESTRATEGIA})...",
                    True,
                    self.TEXTO,
                ),
                (140, 210),
            )
            self.screen.blit(
                self.font_body.render(
                    "Cuando llegue al 100%, veras la carrera completa a velocidad normal (30 FPS).",
                    True,
                    self.MUTED,
                ),
                (140, 255),
            )

            pg.draw.rect(self.screen, self.PANEL_BG, barra_bg, border_radius=8)
            pg.draw.rect(self.screen, self.BORDE, barra_bg, 2, border_radius=8)
            fill_w = max(0, int((barra_bg.w - 4) * (pct / 100.0)))
            if fill_w > 0:
                pg.draw.rect(
                    self.screen,
                    self.ACCENT,
                    pg.Rect(barra_bg.x + 2, barra_bg.y + 2, fill_w, barra_bg.h - 4),
                    border_radius=6,
                )
            pct_txt = self.font_h2.render(f"{pct:.0f}%", True, self.TEXTO)
            self.screen.blit(pct_txt, (barra_bg.centerx - pct_txt.get_width() // 2, barra_bg.y + 4))

            self.screen.blit(
                self.font_body.render(
                    f"Frames: {n_frames} | Progreso pista: {p_pista:.1f}% | Tiempo fisico: {t_fis:.1f} / {TIEMPO_LIMITE_PRUEBA:.0f}s",
                    True,
                    self.TEXTO,
                ),
                (140, 415),
            )
            self.screen.blit(
                self.font_body.render(
                    f"Ultimo RTT: {rtt:.0f} ms (GPU: {gpu:.0f} ms)" + (f" | Aviso: {err[:38]}" if err else ""),
                    True,
                    self.BAD if err else self.MUTED,
                ),
                (140, 450),
            )
            self.screen.blit(
                self.font_small.render("ESC: Cancelar y volver al menu", True, self.MUTED),
                (140, 520),
            )
            pg.display.flip()
            self.clock.tick(30)

        # Fase 2: Reproducción fluida a 30 FPS con todos los datos reales
        idx = 0
        pausado = False
        total_f = len(frames_grabados)
        while total_f > 0:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return None
                    if event.key == pg.K_SPACE:
                        pausado = not pausado
                    elif event.key == pg.K_r:
                        idx = 0
                        pausado = False
                    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        idx = total_f
                        break

            if idx >= total_f:
                break

            snap = frames_grabados[idx]
            e_snap = snap["est"]
            sensores = snap["sensores"]
            impactos = snap["impactos"]
            acc = snap["accion"]

            self._dibujar_pista_y_carro(clave_pista, e_snap, sensores, impactos)
            lineas = [
                (titulo_modo, self.ACCENT),
                (f"Pista: {PISTAS[clave_pista]['nombre']}", self.TEXTO),
                (f"Estrategia: {cfg.ESTRATEGIA} (Replay 100%)", self.WARN),
                (f"Progreso: {snap['progreso_pct']:.1f}%", self.OK),
                (f"Tiempo fisico: {e_snap.tiempo_carrera:.2f} / {TIEMPO_LIMITE_PRUEBA:.0f}s", self.TEXTO),
                (f"Latencia acum: +{snap['lat_acum_s']:.2f}s", self.WARN),
                (f"Reloj + latencia: {e_snap.tiempo_carrera + snap['lat_acum_s']:.2f}s", self.TEXTO),
                (f"RTT: {snap['rtt_ms']:.1f} ms (GPU {snap['gpu_ms']:.1f} ms)", self.MUTED),
                (f"Velocidad: {e_snap.velocidad:.2f} m/s", self.TEXTO),
                (f"Choques: {e_snap.colisiones_obstaculos} | Fuera: {e_snap.tiempo_fuera:.1f}s", self.BAD if e_snap.colisiones_obstaculos else self.MUTED),
                (f"Accion: {ACCIONES_DISCRETAS[acc][0]}", self.ACCENT),
                (f"Frame: {idx + 1}/{total_f}" + (" [PAUSA]" if pausado else ""), self.MUTED),
                ("SPACE: Pausa | R: Repetir | ENTER: Saltar", self.MUTED),
                ("ESC: Abortar y volver al menu", self.MUTED),
            ]
            self._dibujar_panel_lateral(
                lineas,
                sensores,
                velocidad_norm=velocidad_normalizada(e_snap.velocidad),
                historial=snap["hist"],
                accion_texto=ACCIONES_DISCRETAS[acc][0],
                cp_dist=snap["cp_dist_ang"],
            )
            pg.display.flip()
            self.clock.tick(30)
            if not pausado:
                idx += 1

        lat_acum_s = round(sum(latencias_rtt_ms) / 1000.0, 4)
        lat_prom_ms = round(sum(latencias_rtt_ms) / max(1, len(latencias_rtt_ms)), 2)
        tray = [
            [round(f["est"].x, 2), round(f["est"].y, 2), round(f["est"].angulo, 3), int(f["accion"])]
            for f in frames_grabados
        ]
        return {
            "pista": clave_pista,
            "progreso_pct": calcular_progreso_pct(est, clave_pista),
            "tiempo_fisico_s": round(est.tiempo_carrera, 3),
            "latencia_acumulada_s": lat_acum_s,
            "latencia_promedio_ms": lat_prom_ms,
            "completada": est.vueltas >= VUELTAS_OBJETIVO,
            "choques": est.colisiones_obstaculos,
            "tiempo_fuera_s": round(est.tiempo_fuera, 3),
            "trayectoria": tray,
        }

    def _reproducir_frames_autonomo(
        self,
        clave_pista: str,
        frames_grabados: List[Dict[str, Any]],
        titulo_modo: str,
        estrategia_lbl: str,
    ) -> None:
        pg = self.pygame
        idx = 0
        pausado = False
        total_f = len(frames_grabados)
        while total_f > 0:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return
                    if event.key == pg.K_SPACE:
                        pausado = not pausado
                    elif event.key == pg.K_r:
                        idx = 0
                        pausado = False
                    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        return

            if idx >= total_f:
                return

            snap = frames_grabados[idx]
            e_snap = snap["est"]
            sensores = snap["sensores"]
            impactos = snap["impactos"]
            acc = snap["accion"]

            self._dibujar_pista_y_carro(clave_pista, e_snap, sensores, impactos)
            lineas = [
                (titulo_modo, self.ACCENT),
                (f"Pista: {PISTAS[clave_pista]['nombre']}", self.TEXTO),
                (f"Estrategia: {estrategia_lbl} (Replay 100%)", self.WARN),
                (f"Progreso: {snap['progreso_pct']:.1f}%", self.OK),
                (f"Tiempo fisico: {e_snap.tiempo_carrera:.2f} / {TIEMPO_LIMITE_PRUEBA:.0f}s", self.TEXTO),
                (f"Latencia acum: +{snap['lat_acum_s']:.2f}s", self.WARN),
                (f"Reloj + latencia: {e_snap.tiempo_carrera + snap['lat_acum_s']:.2f}s", self.TEXTO),
                (f"RTT: {snap['rtt_ms']:.1f} ms (GPU {snap['gpu_ms']:.1f} ms)", self.MUTED),
                (f"Velocidad: {e_snap.velocidad:.2f} m/s", self.TEXTO),
                (f"Choques: {e_snap.colisiones_obstaculos} | Fuera: {e_snap.tiempo_fuera:.1f}s", self.BAD if e_snap.colisiones_obstaculos else self.MUTED),
                (f"Accion: {ACCIONES_DISCRETAS[acc][0]}", self.ACCENT),
                (f"Frame: {idx + 1}/{total_f}" + (" [PAUSA]" if pausado else ""), self.MUTED),
                ("SPACE: Pausa | R: Repetir | ENTER/ESC: Volver", self.MUTED),
            ]
            self._dibujar_panel_lateral(
                lineas,
                sensores,
                velocidad_norm=velocidad_normalizada(e_snap.velocidad),
                historial=snap["hist"],
                accion_texto=ACCIONES_DISCRETAS[acc][0],
                cp_dist=snap["cp_dist_ang"],
            )
            pg.display.flip()
            self.clock.tick(30)
            if not pausado:
                idx += 1

    def _cancelar_cola_autonoma(self) -> None:
        self._bg_cancel_event.set()
        with self._bg_lock:
            self._bg_queue.clear()

    def _encolar_pistas_autonomas(self, pistas: List[str], jugadas_ref: List[Dict[str, Any]]) -> None:
        with self._bg_lock:
            for p in pistas:
                if p != self._bg_active_track and p not in self._bg_queue:
                    self._bg_results.pop(p, None)
                    self._bg_queue.append(p)
            ya_corriendo = self._bg_thread is not None and self._bg_thread.is_alive()

        if not ya_corriendo:
            self._bg_cancel_event.clear()
            estrategia_snap = str(cfg.ESTRATEGIA)
            icl_snap = copy.deepcopy(cfg.CONFIG_ICL)
            datos_snap = copy.deepcopy(cfg.CONFIG_DATOS)
            ventana = max(1, min(8, int(datos_snap.get("ventana_historia", 5))))

            def _worker_cola() -> None:
                while not self._bg_cancel_event.is_set():
                    with self._bg_lock:
                        if not self._bg_queue:
                            self._bg_active_track = None
                            return
                        clave_pista = self._bg_queue.pop(0)
                        self._bg_active_track = clave_pista
                        self._bg_active_pct = 0.0
                        self._bg_active_frames = 0
                        self._bg_active_pista_pct = 0.0
                        self._bg_active_rtt = 0.0
                        self._bg_active_estrategia = estrategia_snap

                    est = estado_inicial(clave_pista)
                    hist: List[int] = [0] * ventana
                    latencias_rtt_ms: List[float] = []
                    frames_grabados: List[Dict[str, Any]] = []
                    accion_actual = 0
                    frame = 0
                    ultimo_gpu_ms = 0.0
                    ultimo_rtt_ms = 0.0
                    ultimo_err = ""

                    while (
                        not self._bg_cancel_event.is_set()
                        and est.tiempo_carrera < TIEMPO_LIMITE_PRUEBA
                        and est.vueltas < VUELTAS_OBJETIVO
                    ):
                        sensores, impactos = calcular_sensores_rayos(est, clave_pista)
                        cp_dist_ang = info_checkpoint(est, clave_pista)
                        if frame % DECIDE_EVERY_N_FRAMES == 0:
                            resp = self.bridge.decidir_accion(
                                estrategia=estrategia_snap,
                                estado_actual=sensores,
                                historial=hist[-ventana:],
                                jugadas_referencia=jugadas_ref,
                                config_icl=icl_snap,
                                config_datos=datos_snap,
                                velocidad=velocidad_normalizada(est.velocidad),
                                cp_dist=cp_dist_ang[0],
                                cp_ang=cp_dist_ang[1],
                                adapter_name=self.lora_activo,
                            )
                            accion_actual = int(resp["accion"])
                            ultimo_rtt_ms = float(resp["rtt_ms"])
                            ultimo_gpu_ms = float(resp["latencia_gpu_ms"])
                            ultimo_err = str(resp.get("error") or "")
                            latencias_rtt_ms.append(ultimo_rtt_ms)

                        lat_acum_s = sum(latencias_rtt_ms) / 1000.0
                        frames_grabados.append({
                            "est": copy.deepcopy(est),
                            "sensores": list(sensores),
                            "impactos": list(impactos),
                            "accion": accion_actual,
                            "hist": list(hist[-ventana:]),
                            "cp_dist_ang": cp_dist_ang,
                            "lat_acum_s": lat_acum_s,
                            "rtt_ms": ultimo_rtt_ms,
                            "gpu_ms": ultimo_gpu_ms,
                            "err": ultimo_err,
                            "progreso_pct": calcular_progreso_pct(est, clave_pista),
                        })
                        paso_simulacion(est, accion_actual, clave_pista)
                        hist.append(accion_actual)
                        frame += 1

                        prog_pista = calcular_progreso_pct(est, clave_pista)
                        pct_tiempo = (est.tiempo_carrera / TIEMPO_LIMITE_PRUEBA) * 100.0
                        pct_total = 100.0 if est.vueltas >= VUELTAS_OBJETIVO else min(99.0, pct_tiempo)
                        with self._bg_lock:
                            self._bg_active_pct = pct_total
                            self._bg_active_frames = frame
                            self._bg_active_pista_pct = prog_pista
                            self._bg_active_rtt = ultimo_rtt_ms

                    if not self._bg_cancel_event.is_set():
                        resumen = {
                            "pista": clave_pista,
                            "progreso_pct": calcular_progreso_pct(est, clave_pista),
                            "tiempo_fisico_s": round(est.tiempo_carrera, 3),
                            "latencia_acumulada_s": round(sum(latencias_rtt_ms) / 1000.0, 4),
                            "latencia_promedio_ms": round(sum(latencias_rtt_ms) / max(1, len(latencias_rtt_ms)), 2),
                            "completada": est.vueltas >= VUELTAS_OBJETIVO,
                            "choques": est.colisiones_obstaculos,
                            "tiempo_fuera_s": round(est.tiempo_fuera, 3),
                        }
                        with self._bg_lock:
                            self._bg_results[clave_pista] = {
                                "frames": frames_grabados,
                                "resumen": resumen,
                                "estrategia": estrategia_snap,
                            }

                with self._bg_lock:
                    self._bg_active_track = None

            self._bg_thread = threading.Thread(target=_worker_cola, daemon=True)
            self._bg_thread.start()

    def _modo_prueba_a(self) -> None:
        pg = self.pygame
        jugadas_ref = cargar_jugadas()
        if cfg.ESTRATEGIA.upper().strip() == "ICL" and not jugadas_ref:
            self.estado_menu_msg = "Error: graba jugadas en [1] antes de correr ICL."
            return

        btn_p1 = pg.Rect(90, 145, 140, 40)
        btn_p2 = pg.Rect(245, 145, 140, 40)
        btn_p3 = pg.Rect(400, 145, 140, 40)
        btn_todas = pg.Rect(560, 145, 180, 40)
        btn_cancelar_todo = pg.Rect(self.ANCHO - 260, 145, 170, 40)
        btn_volver = pg.Rect(self.ANCHO - 270, 675, 180, 44)

        while True:
            with self._bg_lock:
                act = self._bg_active_track
                act_pct = self._bg_active_pct
                act_frames = self._bg_active_frames
                act_pista_pct = self._bg_active_pista_pct
                act_rtt = self._bg_active_rtt
                act_est = self._bg_active_estrategia
                cola = list(self._bg_queue)
                resultados = dict(self._bg_results)

            btns_accion_fila: Dict[str, Tuple[str, Any]] = {}
            for idx_p, clave in enumerate(PISTAS_TRAIN):
                y_card = 225 + idx_p * 135
                btn_row = pg.Rect(self.ANCHO - 290, y_card + 36, 175, 40)
                if act == clave:
                    btns_accion_fila[clave] = ("cancelar", btn_row)
                elif clave in cola:
                    btns_accion_fila[clave] = ("quitar", btn_row)
                elif clave in resultados:
                    btns_accion_fila[clave] = ("replay", btn_row)
                else:
                    btns_accion_fila[clave] = ("encolar", btn_row)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    return
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if btn_volver.collidepoint(mx, my):
                        return
                    if btn_p1.collidepoint(mx, my):
                        self._encolar_pistas_autonomas(["P1"], jugadas_ref)
                    elif btn_p2.collidepoint(mx, my):
                        self._encolar_pistas_autonomas(["P2"], jugadas_ref)
                    elif btn_p3.collidepoint(mx, my):
                        self._encolar_pistas_autonomas(["P3"], jugadas_ref)
                    elif btn_todas.collidepoint(mx, my):
                        self._encolar_pistas_autonomas(list(PISTAS_TRAIN), jugadas_ref)
                    elif (act or cola) and btn_cancelar_todo.collidepoint(mx, my):
                        self._cancelar_cola_autonoma()
                    else:
                        for clave, (accion_btn, rect_btn) in btns_accion_fila.items():
                            if rect_btn.collidepoint(mx, my):
                                if accion_btn == "cancelar":
                                    self._cancelar_cola_autonoma()
                                elif accion_btn == "quitar":
                                    with self._bg_lock:
                                        if clave in self._bg_queue:
                                            self._bg_queue.remove(clave)
                                elif accion_btn == "replay":
                                    res_info = resultados[clave]
                                    self._reproducir_frames_autonomo(
                                        clave,
                                        res_info["frames"],
                                        "MODO [3]: PRUEBA AUTONOMA",
                                        res_info["estrategia"],
                                    )
                                elif accion_btn == "encolar":
                                    self._encolar_pistas_autonomas([clave], jugadas_ref)
                                break

            self.screen.fill(self.BG)
            self.screen.blit(self.font_title.render("Modo [3]: Simulacion Autonoma en Cola", True, self.ACCENT), (90, 42))
            self.screen.blit(
                self.font_body.render(
                    f"Estrategia actual: {cfg.ESTRATEGIA} | Elige pista para simular en segundo plano y puedes volver al menu.",
                    True,
                    self.MUTED,
                ),
                (90, 86),
            )

            self._dibujar_boton(btn_p1, "+ Simular P1", (24, 38, 62), self.ACCENT)
            self._dibujar_boton(btn_p2, "+ Simular P2", (24, 38, 62), self.ACCENT)
            self._dibujar_boton(btn_p3, "+ Simular P3", (24, 38, 62), self.ACCENT)
            self._dibujar_boton(btn_todas, "+ Simular Todas", (15, 52, 48), self.OK)
            if act or cola:
                self._dibujar_boton(btn_cancelar_todo, "Cancelar Todo", (56, 20, 26), self.BAD)

            for idx_p, clave in enumerate(PISTAS_TRAIN):
                y_card = 225 + idx_p * 135
                card = pg.Rect(90, y_card, self.ANCHO - 180, 114)
                pg.draw.rect(self.screen, self.PANEL_BG, card, border_radius=12)
                pg.draw.rect(self.screen, self.ACCENT if act == clave else self.BORDE, card, 2, border_radius=12)

                nombre_p = PISTAS[clave]["nombre"]
                self.screen.blit(self.font_h2.render(nombre_p, True, self.TEXTO), (card.x + 22, card.y + 14))

                accion_btn, rect_btn = btns_accion_fila[clave]
                if act == clave:
                    self.screen.blit(
                        self.font_body.render(
                            f"Simulando ({act_est})... {act_pct:.0f}% | Frames: {act_frames} | Pista: {act_pista_pct:.1f}% | RTT: {act_rtt:.0f} ms",
                            True,
                            self.WARN,
                        ),
                        (card.x + 22, card.y + 46),
                    )
                    barra = pg.Rect(card.x + 22, card.y + 76, card.w - 250, 18)
                    pg.draw.rect(self.screen, self.BG, barra, border_radius=5)
                    fw = max(0, int((barra.w - 4) * (act_pct / 100.0)))
                    if fw > 0:
                        pg.draw.rect(self.screen, self.ACCENT, pg.Rect(barra.x + 2, barra.y + 2, fw, barra.h - 4), border_radius=4)
                    self._dibujar_boton(rect_btn, "Cancelar", (56, 20, 26), self.BAD)
                elif clave in cola:
                    pos_q = cola.index(clave) + 1
                    self.screen.blit(
                        self.font_body.render(f"En cola (turno #{pos_q}) esperando servidor...", True, self.MUTED),
                        (card.x + 22, card.y + 52),
                    )
                    self._dibujar_boton(rect_btn, "Quitar de cola", (56, 38, 16), self.WARN)
                elif clave in resultados:
                    res = resultados[clave]["resumen"]
                    est_r = resultados[clave]["estrategia"]
                    col_r = self.OK if res["completada"] else self.WARN
                    self.screen.blit(
                        self.font_body.render(
                            f"Listo ({est_r}): Progreso {res['progreso_pct']:.1f}% | t={res['tiempo_fisico_s']:.1f}s | "
                            f"Choques={res['choques']} | Lat={res['latencia_acumulada_s']:.1f}s",
                            True,
                            col_r,
                        ),
                        (card.x + 22, card.y + 52),
                    )
                    self._dibujar_boton(rect_btn, "Ver Replay (30 FPS)", (15, 52, 48), self.OK)
                else:
                    self.screen.blit(
                        self.font_body.render("Sin simular aun. Pulsa '+ Simular' para enviar a la cola.", True, self.MUTED),
                        (card.x + 22, card.y + 52),
                    )
                    self._dibujar_boton(rect_btn, f"Simular {clave}", (24, 38, 62), self.ACCENT)

            self._dibujar_boton(btn_volver, "Volver al Menu", self.PANEL_BG, self.BORDE)
            self.screen.blit(
                self.font_small.render(
                    "Puedes volver al menu mientras carga para editar Configuracion o Grabar Jugadas.",
                    True,
                    self.MUTED,
                ),
                (90, 688),
            )
            pg.display.flip()
            self.clock.tick(30)

    def _pantalla_carrera_todos(self) -> None:
        pg = self.pygame
        try:
            datos = self.bridge.obtener_trayectorias_b().get("pistas", {})
        except Exception as exc:
            self.estado_menu_msg = f"Error cargando carreras: {exc}"
            return

        paleta = [
            (45, 212, 191),
            (250, 204, 21),
            (129, 140, 248),
            (248, 113, 113),
            (74, 222, 128),
            (251, 146, 60),
            (244, 114, 182),
            (56, 189, 248),
            (192, 132, 252),
            (163, 230, 53),
            (253, 186, 116),
            (94, 234, 212),
        ]
        pista_sel = "B1"
        modo_legendario = False
        pausado = False
        frame_idx = 0
        estados_leg: List[Dict[str, Any]] = []

        def _iniciar_pista(clave: str) -> None:
            nonlocal frame_idx, estados_leg
            frame_idx = 0
            estados_leg = []
            for idx_e, item in enumerate(datos.get(clave, [])):
                est_c = estado_inicial(clave)
                estados_leg.append({
                    "nombre": f"{item.get('student_name', '?')} ({item.get('estrategia', '-')})",
                    "color": paleta[idx_e % len(paleta)],
                    "tray": item.get("trayectoria", []),
                    "est": est_c,
                    "choques_pvp": 0,
                })

        _iniciar_pista(pista_sel)

        while True:
            x0 = self.AREA_PISTA_W
            btn_b1 = pg.Rect(x0 + 18, 56, 76, 30)
            btn_b2 = pg.Rect(x0 + 102, 56, 76, 30)
            btn_b3 = pg.Rect(x0 + 186, 56, 76, 30)
            btn_modo = pg.Rect(x0 + 18, 96, 310, 34)
            btn_reset = pg.Rect(x0 + 18, 138, 150, 30)
            btn_volver = pg.Rect(x0 + 178, 138, 150, 30)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return
                    if event.key == pg.K_SPACE:
                        pausado = not pausado
                    elif event.key == pg.K_r:
                        _iniciar_pista(pista_sel)
                    elif event.key == pg.K_m:
                        modo_legendario = not modo_legendario
                        _iniciar_pista(pista_sel)
                    elif event.key == pg.K_1:
                        pista_sel = "B1"
                        _iniciar_pista(pista_sel)
                    elif event.key == pg.K_2:
                        pista_sel = "B2"
                        _iniciar_pista(pista_sel)
                    elif event.key == pg.K_3:
                        pista_sel = "B3"
                        _iniciar_pista(pista_sel)
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if btn_b1.collidepoint(pos):
                        pista_sel = "B1"
                        _iniciar_pista(pista_sel)
                    elif btn_b2.collidepoint(pos):
                        pista_sel = "B2"
                        _iniciar_pista(pista_sel)
                    elif btn_b3.collidepoint(pos):
                        pista_sel = "B3"
                        _iniciar_pista(pista_sel)
                    elif btn_modo.collidepoint(pos):
                        modo_legendario = not modo_legendario
                        _iniciar_pista(pista_sel)
                    elif btn_reset.collidepoint(pos):
                        _iniciar_pista(pista_sel)
                    elif btn_volver.collidepoint(pos):
                        return

            # Dibujar pista y obstáculos
            self.screen.fill(self.BG)
            puntos = PISTAS[pista_sel]["puntos"]
            ancho = PISTAS[pista_sel]["ancho"]
            escala_ref = self._w2s(0.0, 0.0)[2]
            poly = [self._w2s(px, py)[:2] for px, py in puntos]
            if len(poly) >= 2:
                pg.draw.lines(self.screen, (42, 52, 65), True, poly, max(8, int(ancho * 2.0 * escala_ref)))
                pg.draw.lines(self.screen, (71, 85, 105), True, poly, 2)
            mx0, my0, _ = self._w2s(puntos[0][0], puntos[0][1])
            pg.draw.circle(self.screen, self.WARN, (mx0, my0), 7)

            t_act = frame_idx * DT
            for ox, oy, rad in obtener_obstaculos_pista(pista_sel, t_act):
                osx, osy, _ = self._w2s(ox, oy)
                pg.draw.circle(self.screen, self.BAD, (osx, osy), max(4, int(rad * escala_ref)))

            # Avanzar física si está en Modo Legendario
            max_frames = max((len(c["tray"]) for c in estados_leg), default=0)
            if not pausado and frame_idx < max_frames:
                if modo_legendario:
                    for c in estados_leg:
                        tray = c["tray"]
                        est_c = c["est"]
                        if est_c.vueltas < VUELTAS_OBJETIVO and tray:
                            f_use = min(frame_idx, len(tray) - 1)
                            acc = int(tray[f_use][3]) if len(tray[f_use]) >= 4 else 0
                            paso_simulacion(est_c, acc, pista_sel)
                    # Colisiones elásticas entre carros (hitbox activo)
                    n_c = len(estados_leg)
                    for i in range(n_c):
                        for j in range(i + 1, n_c):
                            ei = estados_leg[i]["est"]
                            ej = estados_leg[j]["est"]
                            dx = ej.x - ei.x
                            dy = ej.y - ei.y
                            dist = math.hypot(dx, dy)
                            min_d = RADIO_CARRO * 2.1
                            if dist < min_d:
                                nx = dx / dist if dist > 1e-6 else 1.0
                                ny = dy / dist if dist > 1e-6 else 0.0
                                push = (min_d - dist) * 0.55
                                ei.x -= nx * push
                                ei.y -= ny * push
                                ej.x += nx * push
                                ej.y += ny * push
                                ei.velocidad *= 0.85
                                ej.velocidad *= 0.85
                                estados_leg[i]["choques_pvp"] += 1
                                estados_leg[j]["choques_pvp"] += 1
                frame_idx += 1

            # Dibujar todos los carros superpuestos
            for c in estados_leg:
                tray = c["tray"]
                if not tray:
                    continue
                if modo_legendario:
                    cx, cy, cang = c["est"].x, c["est"].y, c["est"].angulo
                else:
                    f_use = min(frame_idx, len(tray) - 1)
                    cx, cy, cang = float(tray[f_use][0]), float(tray[f_use][1]), float(tray[f_use][2])

                sx, sy, esc = self._w2s(cx, cy)
                col = c["color"]
                pts_tri = [
                    (sx + math.cos(cang) * 0.42 * esc, sy + math.sin(cang) * 0.42 * esc),
                    (sx + math.cos(cang + 2.4) * 0.30 * esc, sy + math.sin(cang + 2.4) * 0.30 * esc),
                    (sx + math.cos(cang - 2.4) * 0.30 * esc, sy + math.sin(cang - 2.4) * 0.30 * esc),
                ]
                pg.draw.polygon(self.screen, col, pts_tri)
                lbl_c = self.font_small.render(c["nombre"][:14], True, col)
                self.screen.blit(lbl_c, (sx - lbl_c.get_width() // 2, sy - 20))

            # Panel lateral de control
            pg.draw.rect(self.screen, self.PANEL_BG, (x0, 0, self.ANCHO - x0, self.ALTO))
            pg.draw.line(self.screen, self.BORDE, (x0, 0), (x0, self.ALTO), 2)
            self.screen.blit(self.font_h2.render(f"Carrera Todos — {pista_sel}", True, self.ACCENT), (x0 + 18, 18))
            self._dibujar_boton(btn_b1, "B1", (24, 38, 62) if pista_sel == "B1" else self.BG, self.ACCENT if pista_sel == "B1" else self.BORDE)
            self._dibujar_boton(btn_b2, "B2", (24, 38, 62) if pista_sel == "B2" else self.BG, self.ACCENT if pista_sel == "B2" else self.BORDE)
            self._dibujar_boton(btn_b3, "B3", (24, 38, 62) if pista_sel == "B3" else self.BG, self.ACCENT if pista_sel == "B3" else self.BORDE)

            if modo_legendario:
                self._dibujar_boton(btn_modo, "MODO LEGENDARIO (Choques ON)", (68, 22, 28), self.BAD)
            else:
                self._dibujar_boton(btn_modo, "Modo Fantasma (Transparente)", (15, 52, 48), self.ACCENT)
            self._dibujar_boton(btn_reset, "Reiniciar (R)", self.BG, self.BORDE)
            self._dibujar_boton(btn_volver, "Volver (ESC)", self.BG, self.BORDE)

            self.screen.blit(
                self.font_small.render(f"Frame {frame_idx}/{max_frames}" + (" [PAUSA]" if pausado else ""), True, self.MUTED),
                (x0 + 18, 180),
            )
            if not estados_leg:
                self.screen.blit(
                    self.font_body.render("Aun no hay trayectorias en esta pista.", True, self.WARN),
                    (x0 + 18, 220),
                )
            else:
                y_l = 210
                for c in estados_leg[:14]:
                    extra = f" | PvP:{c['choques_pvp']}" if modo_legendario else ""
                    self.screen.blit(
                        self.font_small.render(f"{c['nombre'][:18]}{extra}", True, c["color"]),
                        (x0 + 18, y_l),
                    )
                    y_l += 24

            self.screen.blit(
                self.font_small.render("SPACE: Pausa | M: Fantasma/Legendario", True, self.MUTED),
                (x0 + 18, self.ALTO - 34),
            )
            pg.display.flip()
            self.clock.tick(30)

    def _modo_evaluacion_b(self) -> None:
        jugadas_ref = cargar_jugadas()
        if cfg.ESTRATEGIA.upper().strip() in {"ICL", "LORA_ICL"} and not jugadas_ref:
            self.estado_menu_msg = f"Error: graba jugadas en [1] antes de correr evaluacion {cfg.ESTRATEGIA}."
            return
        resultados: Dict[str, Dict[str, Any]] = {}
        for clave in PISTAS_TEST:
            r = self._correr_pista_autonoma_visual(clave, jugadas_ref, "MODO [4]: EVALUACION OFICIAL")
            if r is None:
                self.estado_menu_msg = "Evaluacion oficial cancelada (no enviada)."
                return
            resultados[clave] = r

        try:
            num_ej = (
                len(jugadas_ref)
                if cfg.ESTRATEGIA.upper().strip() == "ICL"
                else min(len(jugadas_ref), int(cfg.CONFIG_LORA.get("num_ejemplos", 200)))
            )
            resp = self.bridge.registrar_puntaje(cfg.ESTRATEGIA, resultados, num_ejemplos=num_ej)
            reg = resp["registro"]
            pts_lbl = reg.get("puntos_f1", reg.get("score_final", 0.0))
            self.estado_menu_msg = (
                f"Ranking #{resp['posicion']}/{resp['total_estudiantes']} | "
                f"Pts F1={pts_lbl} | %B={reg['progreso_promedio_pct']}%"
            )
        except Exception as exc:
            self.estado_menu_msg = f"Error al enviar puntaje: {exc}"

    def run(self) -> None:
        try:
            while True:
                modo = self._pantalla_menu()
                if modo is None:
                    break
                if modo == 1:
                    self._modo_grabar_jugadas()
                elif modo == 2:
                    self._modo_entrenar_lora()
                elif modo == 3:
                    self._modo_prueba_a()
                elif modo == 4:
                    self._modo_evaluacion_b()
        finally:
            self.pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cliente Simulador UIS Grand Prix")
    parser.add_argument(
        "--modo",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Ejecuta directamente un modo sin pasar por el menú (2 y 4 soportan --headless).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Ejecuta modos 2, 3 o 4 en consola sin abrir ventana Pygame.",
    )
    parser.add_argument(
        "--generar-semilla",
        action="store_true",
        help="Genera data/mis_jugadas.json de demostración usando el piloto guía.",
    )
    args = parser.parse_args()

    if args.generar_semilla:
        muestras = generar_jugadas_semilla()
        guardar_jugadas(muestras)
        print(f"Generadas {len(muestras)} jugadas de demostración en {JUGADAS_PATH}")
        return

    bridge = ServerBridge()

    if args.headless:
        if args.modo == 2:
            ejecutar_entrenamiento_lora(bridge)
        elif args.modo == 3:
            jugadas_ref = cargar_jugadas()
            for clave in PISTAS_TRAIN:
                r = simular_pista_autonoma_headless(bridge, clave, jugadas_ref)
                print(f"[Modo 3] {clave}: {r}")
        elif args.modo == 4:
            ejecutar_evaluacion_oficial_headless(bridge)
        else:
            print("Con --headless especifica --modo 2, 3 o 4.")
        return

    app = JuegoPygame(bridge)
    if args.modo == 1:
        app._modo_grabar_jugadas()
        app.pygame.quit()
    elif args.modo == 2:
        app._modo_entrenar_lora()
        print(app.estado_menu_msg)
        app.pygame.quit()
    elif args.modo == 3:
        app._modo_prueba_a()
        print(app.estado_menu_msg)
        app.pygame.quit()
    elif args.modo == 4:
        app._modo_evaluacion_b()
        print(app.estado_menu_msg)
        app.pygame.quit()
    else:
        app.run()


if __name__ == "__main__":
    main()
