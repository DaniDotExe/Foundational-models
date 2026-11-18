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
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

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
RECORD_EVERY_N_FRAMES = max(1, int(os.getenv("RECORD_EVERY_N_FRAMES", "6")))

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

    # Límite del recuadro negro: si choca contra el borde exterior, rebota mirando al checkpoint
    limite_x, limite_y = 11.0, 9.9
    if abs(e.x) >= limite_x or abs(e.y) >= limite_y:
        e.x = limitar(e.x, -limite_x, limite_x)
        e.y = limitar(e.y, -limite_y, limite_y)
        tx_cp, ty_cp = puntos[e.siguiente_checkpoint]
        e.angulo = envolver_angulo(math.atan2(ty_cp - e.y, tx_cp - e.x))
        e.velocidad *= 0.40

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


def info_checkpoint(estado: EstadoCarro, clave_pista: str) -> float:
    """Devuelve la distancia normalizada [0..1] al próximo checkpoint."""
    puntos = PISTAS[clave_pista]["puntos"]
    tx, ty = puntos[estado.siguiente_checkpoint % len(puntos)]
    dx, dy = tx - estado.x, ty - estado.y
    return round(limitar(math.hypot(dx, dy) / 12.0, 0.0, 1.0), 4)


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
                muestras.append({
                    "pista": clave,
                    "sensores": sensores,
                    "velocidad": velocidad_normalizada(est.velocidad),
                    "cp_dist": info_checkpoint(est, clave),
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

    while est.tiempo_carrera < TIEMPO_LIMITE_PRUEBA and est.vueltas < VUELTAS_OBJETIVO:
        if frame % RECORD_EVERY_N_FRAMES == 0:
            sensores, _ = calcular_sensores_rayos(est, clave_pista)
            resp = bridge.decidir_accion(
                estrategia=cfg.ESTRATEGIA,
                estado_actual=sensores,
                historial=hist[-ventana:],
                jugadas_referencia=jugadas_ref,
                config_icl=cfg.CONFIG_ICL,
                config_datos=cfg.CONFIG_DATOS,
                velocidad=velocidad_normalizada(est.velocidad),
                cp_dist=info_checkpoint(est, clave_pista),
            )
            accion_actual = int(resp["accion"])
            latencias_rtt_ms.append(float(resp["rtt_ms"]))
            latencias_gpu_ms.append(float(resp["latencia_gpu_ms"]))

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

    respuesta = bridge.registrar_puntaje(cfg.ESTRATEGIA, resultados)
    reg = respuesta["registro"]
    print(
        f"\n[RANKING] Posición #{respuesta['posicion']}/{respuesta['total_estudiantes']} | "
        f"Score Final: {reg['score_final']} | % Promedio B: {reg['progreso_promedio_pct']}%"
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
        cp_dist: Optional[float] = None,
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
            cp_dist_v = limitar(float(cp_dist), 0.0, 1.0)
            cp_color = (56, 189, 248)
            lbl = self.font_small.render(f"{'cp_dist':9s}: {cp_dist_v:.2f}", True, cp_color)
            self.screen.blit(lbl, (x0 + 18, y))
            bar_x = x0 + 150
            bar_w = w - 175
            pg.draw.rect(self.screen, (30, 41, 59), (bar_x, y + 2, bar_w, 12), border_radius=4)
            pg.draw.rect(self.screen, cp_color, (bar_x, y + 2, int(bar_w * cp_dist_v), 12), border_radius=4)
            y += 21

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

            # Tarjeta Izquierda: ICL y Datos
            pg.draw.rect(self.screen, self.PANEL_BG, pg.Rect(90, 148, 490, 360), border_radius=12)
            pg.draw.rect(self.screen, self.BORDE, pg.Rect(90, 148, 490, 360), 2, border_radius=12)
            self.screen.blit(self.font_h2.render("ICL (Sin modificar pesos)", True, self.ACCENT), (115, 154))
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
            pg.draw.rect(self.screen, self.BORDE, pg.Rect(630, 148, 500, 360), 2, border_radius=12)
            self.screen.blit(self.font_h2.render("LoRA (Adapta pesos PEFT)", True, (165, 180, 252)), (655, 154))

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

    def _pantalla_menu(self) -> Optional[int]:
        pg = self.pygame
        botones = [
            (1, "[1] Grabar Jugadas (Pistas P1, P2, P3)", "Conduce con teclado y guarda data/mis_jugadas.json"),
            (2, "[2] Entrenar Piloto (Solo ESTRATEGIA == LORA)", "Envia tus jugadas al servidor para entrenar adaptador PEFT"),
            (3, "[3] Modo Autonomo - Prueba (Pistas P)", "Prueba en vivo tu piloto consultando al servidor"),
            (4, "[4] Evaluacion Oficial (Pistas B1, B2, B3)", "Corre las 3 pistas ocultas y registra tu puntaje al ranking"),
        ]
        btn_editar_nombre = pg.Rect(self.ANCHO - 560, 82, 145, 32)
        btn_icl = pg.Rect(self.ANCHO - 405, 82, 95, 32)
        btn_lora = pg.Rect(self.ANCHO - 302, 82, 95, 32)
        btn_config = pg.Rect(self.ANCHO - 197, 82, 107, 32)
        btn_copiar_cmd = pg.Rect(self.ANCHO - 250, 142, 160, 28)
        cmd_backend = f'cd /d "{BASE_DIR.parent / "backend_server"}" && python -m uvicorn server:app --host 0.0.0.0 --port 8000'

        while True:
            rects = []
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.KEYDOWN:
                    if event.key in (pg.K_1, pg.K_KP1):
                        return 1
                    if event.key in (pg.K_2, pg.K_KP2):
                        return 2
                    if event.key in (pg.K_3, pg.K_KP3):
                        return 3
                    if event.key in (pg.K_4, pg.K_KP4):
                        return 4
                    if event.key == pg.K_h:
                        self._chequear_salud_silenciosa()
                    if event.key == pg.K_ESCAPE:
                        return None
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if btn_editar_nombre.collidepoint((mx, my)):
                        self._pedir_nombre_participante(permitir_cancelar=True)
                    elif btn_icl.collidepoint((mx, my)):
                        cfg.ESTRATEGIA = "ICL"
                        self.estado_menu_msg = "Estrategia cambiada a ICL (sin modificar pesos)."
                    elif btn_lora.collidepoint((mx, my)):
                        cfg.ESTRATEGIA = "LORA"
                        self.estado_menu_msg = "Estrategia cambiada a LORA (usa [2] para entrenar si aun no lo has hecho)."
                    elif btn_config.collidepoint((mx, my)):
                        self._pantalla_configurar()
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
                    for idx, (m_id, _, _) in enumerate(botones):
                        r = pg.Rect(90, 215 + idx * 105, self.ANCHO - 180, 86)
                        if r.collidepoint(mx, my):
                            return m_id

            self.screen.fill(self.BG)
            t1 = self.font_title.render("UIS Grand Prix Foundationals Models", True, self.ACCENT)
            self.screen.blit(t1, (90, 42))

            self.screen.blit(
                self.font_body.render(f"Participante: {self.bridge.student_name} | Estrategia: {cfg.ESTRATEGIA}", True, self.TEXTO),
                (90, 86),
            )
            self._dibujar_boton(btn_editar_nombre, "Editar Nombre", self.PANEL_BG, self.BORDE)
            self._dibujar_boton(
                btn_icl,
                "ICL",
                (15, 52, 48) if cfg.ESTRATEGIA == "ICL" else self.PANEL_BG,
                self.ACCENT if cfg.ESTRATEGIA == "ICL" else self.BORDE,
            )
            self._dibujar_boton(
                btn_lora,
                "LORA",
                (35, 36, 78) if cfg.ESTRATEGIA == "LORA" else self.PANEL_BG,
                (129, 140, 248) if cfg.ESTRATEGIA == "LORA" else self.BORDE,
            )
            self._dibujar_boton(btn_config, "Configurar", (24, 38, 62), self.ACCENT)

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

            for idx, (m_id, titulo, desc) in enumerate(botones):
                r = pg.Rect(90, 215 + idx * 105, self.ANCHO - 180, 86)
                rects.append(r)
                hover = r.collidepoint(pg.mouse.get_pos())
                bg = (24, 36, 58) if hover else self.PANEL_BG
                border = self.ACCENT if hover else self.BORDE
                pg.draw.rect(self.screen, bg, r, border_radius=12)
                pg.draw.rect(self.screen, border, r, 2, border_radius=12)
                self.screen.blit(self.font_h2.render(titulo, True, self.TEXTO), (r.x + 22, r.y + 16))
                self.screen.blit(self.font_body.render(desc, True, self.MUTED), (r.x + 22, r.y + 46))

            msg_c = self.WARN if "Error" in self.estado_menu_msg or "Solo" in self.estado_menu_msg else self.TEXTO
            self.screen.blit(self.font_body.render(f"Estado: {self.estado_menu_msg}", True, msg_c), (90, 660))
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
                self._dibujar_panel_lateral(
                    lineas,
                    sens,
                    velocidad_norm=vel_norm if vel_norm is not None else 0.0,
                    historial=hist,
                    accion_texto=f"{accion} ({nom_acc})",
                    cp_dist=float(cp_d) if cp_d is not None else None,
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
                jugadas.append({
                    "pista": clave,
                    "sensores": sensores,
                    "velocidad": velocidad_normalizada(est.velocidad),
                    "cp_dist": info_checkpoint(est, clave),
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

            self._dibujar_boton(btn_ver_frames, "", (24, 38, 62), self.ACCENT)
            l1 = self.font_body.render(f"Ver Frames: {len(jugadas)}", True, self.TEXTO)
            l2 = self.font_small.render(f"P1: {conteo['P1']}  |  P2: {conteo['P2']}  |  P3: {conteo['P3']}", True, self.ACCENT)
            self.screen.blit(l1, (btn_ver_frames.x + (btn_ver_frames.w - l1.get_width()) // 2, btn_ver_frames.y + 10))
            self.screen.blit(l2, (btn_ver_frames.x + (btn_ver_frames.w - l2.get_width()) // 2, btn_ver_frames.y + 35))
            self._dibujar_boton(btn_guardar_salir, "Guardar y Salir", (15, 52, 48), self.ACCENT)

            pg.display.flip()
            self.clock.tick(30)

    def _modo_entrenar_lora(self) -> None:
        pg = self.pygame
        self.screen.fill(self.BG)
        self.screen.blit(
            self.font_title.render("Entrenando adaptador LoRA en el servidor...", True, self.ACCENT),
            (100, 260),
        )
        self.screen.blit(
            self.font_body.render("Espera unos segundos mientras la GPU ajusta las matrices PEFT.", True, self.MUTED),
            (100, 310),
        )
        pg.display.flip()

        try:
            res = ejecutar_entrenamiento_lora(self.bridge)
            self.estado_menu_msg = (
                f"LoRA entrenado OK ({res['ejemplos_entrenados']} ej, "
                f"loss={res['loss_final']}, t={res['tiempo_entrenamiento_s']}s)."
            )
        except Exception as exc:
            self.estado_menu_msg = f"Error LoRA: {exc}"

    def _correr_pista_autonoma_visual(
        self, clave_pista: str, jugadas_ref: List[Dict[str, Any]], titulo_modo: str
    ) -> Optional[Dict[str, Any]]:
        pg = self.pygame
        est = estado_inicial(clave_pista)
        ventana = max(1, min(8, int(cfg.CONFIG_DATOS.get("ventana_historia", 5))))
        hist: List[int] = [0] * ventana
        accion_actual = 0
        frame = 0
        latencias_rtt_ms: List[float] = []
        ultimo_gpu_ms = 0.0
        ultimo_rtt_ms = 0.0
        ultimo_err = ""

        while est.tiempo_carrera < TIEMPO_LIMITE_PRUEBA and est.vueltas < VUELTAS_OBJETIVO:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    return None

            sensores, impactos = calcular_sensores_rayos(est, clave_pista)

            # Sincronización física: espera la respuesta HTTP en cada paso de consulta
            if frame % RECORD_EVERY_N_FRAMES == 0:
                resp = self.bridge.decidir_accion(
                    estrategia=cfg.ESTRATEGIA,
                    estado_actual=sensores,
                    historial=hist[-ventana:],
                    jugadas_referencia=jugadas_ref,
                    config_icl=cfg.CONFIG_ICL,
                    config_datos=cfg.CONFIG_DATOS,
                    velocidad=velocidad_normalizada(est.velocidad),
                    cp_dist=info_checkpoint(est, clave_pista),
                )
                accion_actual = int(resp["accion"])
                ultimo_rtt_ms = float(resp["rtt_ms"])
                ultimo_gpu_ms = float(resp["latencia_gpu_ms"])
                ultimo_err = str(resp.get("error") or "")
                latencias_rtt_ms.append(ultimo_rtt_ms)

            paso_simulacion(est, accion_actual, clave_pista)
            hist.append(accion_actual)
            frame += 1

            lat_acum_s = sum(latencias_rtt_ms) / 1000.0
            self._dibujar_pista_y_carro(clave_pista, est, sensores, impactos)
            lineas = [
                (titulo_modo, self.ACCENT),
                (f"Pista: {PISTAS[clave_pista]['nombre']}", self.TEXTO),
                (f"Estrategia: {cfg.ESTRATEGIA}", self.WARN),
                (f"Progreso: {calcular_progreso_pct(est, clave_pista):.1f}%", self.OK),
                (f"Tiempo fisico: {est.tiempo_carrera:.2f} / {TIEMPO_LIMITE_PRUEBA:.0f}s", self.TEXTO),
                (f"Latencia acum: +{lat_acum_s:.2f}s", self.WARN),
                (f"Reloj + latencia: {est.tiempo_carrera + lat_acum_s:.2f}s", self.TEXTO),
                (f"RTT ultimo: {ultimo_rtt_ms:.1f} ms (GPU {ultimo_gpu_ms:.1f} ms)", self.MUTED),
                (f"Velocidad: {est.velocidad:.2f} m/s", self.TEXTO),
                (f"Choques: {est.colisiones_obstaculos} | Fuera: {est.tiempo_fuera:.1f}s", self.BAD if est.colisiones_obstaculos else self.MUTED),
                (f"Accion: {ACCIONES_DISCRETAS[accion_actual][0]}", self.ACCENT),
                (f"Aviso: {ultimo_err[:28]}" if ultimo_err else "Conexion OK", self.BAD if ultimo_err else self.OK),
                ("", self.TEXTO),
                ("ESC: Abortar y volver al menu", self.MUTED),
            ]
            self._dibujar_panel_lateral(
                lineas,
                sensores,
                velocidad_norm=velocidad_normalizada(est.velocidad),
                historial=hist[-ventana:],
                accion_texto=ACCIONES_DISCRETAS[accion_actual][0],
                cp_dist=info_checkpoint(est, clave_pista),
            )
            pg.display.flip()
            self.clock.tick(30)

        lat_acum_s = round(sum(latencias_rtt_ms) / 1000.0, 4)
        lat_prom_ms = round(sum(latencias_rtt_ms) / max(1, len(latencias_rtt_ms)), 2)
        return {
            "pista": clave_pista,
            "progreso_pct": calcular_progreso_pct(est, clave_pista),
            "tiempo_fisico_s": round(est.tiempo_carrera, 3),
            "latencia_acumulada_s": lat_acum_s,
            "latencia_promedio_ms": lat_prom_ms,
            "completada": est.vueltas >= VUELTAS_OBJETIVO,
            "choques": est.colisiones_obstaculos,
            "tiempo_fuera_s": round(est.tiempo_fuera, 3),
        }

    def _modo_prueba_a(self) -> None:
        jugadas_ref = cargar_jugadas()
        if cfg.ESTRATEGIA.upper().strip() == "ICL" and not jugadas_ref:
            self.estado_menu_msg = "Error: graba jugadas en [1] antes de correr ICL."
            return
        resumen = []
        for clave in PISTAS_TRAIN:
            r = self._correr_pista_autonoma_visual(clave, jugadas_ref, "MODO [3]: PRUEBA AUTONOMA")
            if r is None:
                self.estado_menu_msg = "Prueba en Pistas A cancelada."
                return
            resumen.append(f"{clave}:{r['progreso_pct']:.0f}%")
        self.estado_menu_msg = f"Prueba Pistas A completada -> {' | '.join(resumen)}"

    def _modo_evaluacion_b(self) -> None:
        jugadas_ref = cargar_jugadas()
        if cfg.ESTRATEGIA.upper().strip() == "ICL" and not jugadas_ref:
            self.estado_menu_msg = "Error: graba jugadas en [1] antes de correr evaluacion ICL."
            return
        resultados: Dict[str, Dict[str, Any]] = {}
        for clave in PISTAS_TEST:
            r = self._correr_pista_autonoma_visual(clave, jugadas_ref, "MODO [4]: EVALUACION OFICIAL")
            if r is None:
                self.estado_menu_msg = "Evaluacion oficial cancelada (no enviada)."
                return
            resultados[clave] = r

        try:
            resp = self.bridge.registrar_puntaje(cfg.ESTRATEGIA, resultados)
            reg = resp["registro"]
            self.estado_menu_msg = (
                f"Ranking #{resp['posicion']}/{resp['total_estudiantes']} | "
                f"Score={reg['score_final']} | %B={reg['progreso_promedio_pct']}%"
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
