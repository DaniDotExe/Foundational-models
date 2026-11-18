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
    "A1": {
        "nombre": "A1 - Óvalo (Entrenamiento)",
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
    "A2": {
        "nombre": "A2 - Curvas Amplias (Entrenamiento)",
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
    "A3": {
        "nombre": "A3 - Chicane Técnica (Entrenamiento)",
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

PISTAS_TRAIN = ["A1", "A2", "A3"]
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
    e = estado

    e.velocidad = limitar(
        e.velocidad + (acel * ACELERACION - freno * FRENO - ARRASTRE * e.velocidad) * DT,
        0.0,
        VELOCIDAD_MAX,
    )
    e.angulo = envolver_angulo(
        e.angulo + giro * GIRO_MAX * (0.28 + 0.72 * e.velocidad / VELOCIDAD_MAX) * DT
    )
    e.x += math.cos(e.angulo) * e.velocidad * DT
    e.y += math.sin(e.angulo) * e.velocidad * DT
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
            if toco_obstaculo or abs(error_lateral_firmado(rx, ry, puntos)) > ancho:
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
    """Heurística de demostración opcional (tecla P en modo grabación)."""
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


def generar_jugadas_semilla(max_por_pista: int = 40) -> List[Dict[str, Any]]:
    """Genera jugadas limpias de demostración (~120 muestras) en A1, A2, A3."""
    muestras: List[Dict[str, Any]] = []
    ventana = max(1, min(4, int(cfg.CONFIG_DATOS.get("ventana_historia", 3))))
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
                    "historial": hist[-ventana:],
                    "accion": accion,
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
    ventana = max(1, min(4, int(cfg.CONFIG_DATOS.get("ventana_historia", 3))))
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
        pg.draw.lines(self.screen, self.ASFALTO, True, pts_screen, ancho_px)

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

        # Rayos de los 5 sensores
        car_x, car_y, _ = self._w2s(estado.x, estado.y)
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

    def _dibujar_panel_lateral(self, lineas: List[Tuple[str, Tuple[int, int, int]]], sensores: List[float]) -> None:
        pg = self.pygame
        x0 = self.AREA_PISTA_W
        w = self.ANCHO - x0
        pg.draw.rect(self.screen, self.PANEL_BG, (x0, 0, w, self.ALTO))
        pg.draw.line(self.screen, self.BORDE, (x0, 0), (x0, self.ALTO), 2)

        y = 20
        for texto, color in lineas:
            surf = self.font_body.render(texto, True, color)
            self.screen.blit(surf, (x0 + 18, y))
            y += 24

        y += 12
        hdr = self.font_h2.render("Sensores (5 Rayos)", True, self.ACCENT)
        self.screen.blit(hdr, (x0 + 18, y))
        y += 28
        for nombre, val in zip(NOMBRES_RAYOS, sensores):
            lbl = self.font_small.render(f"{nombre:9s}: {val:.2f}", True, self.TEXTO)
            self.screen.blit(lbl, (x0 + 18, y))
            bar_x = x0 + 150
            bar_w = w - 175
            pg.draw.rect(self.screen, (30, 41, 59), (bar_x, y + 2, bar_w, 12), border_radius=4)
            fill_c = self.OK if val > 0.5 else (self.WARN if val > 0.25 else self.BAD)
            pg.draw.rect(self.screen, fill_c, (bar_x, y + 2, int(bar_w * val), 12), border_radius=4)
            y += 24

    def _pantalla_menu(self) -> Optional[int]:
        pg = self.pygame
        botones = [
            (1, "[1] Grabar Jugadas (Pistas A1, A2, A3)", "Conduce con teclado y guarda data/mis_jugadas.json"),
            (2, "[2] Entrenar Piloto (Solo ESTRATEGIA == LORA)", "Envia tus jugadas al servidor para entrenar adaptador PEFT"),
            (3, "[3] Modo Autonomo - Prueba (Pistas A)", "Prueba en vivo tu piloto consultando al servidor"),
            (4, "[4] Evaluacion Oficial (Pistas B1, B2, B3)", "Corre las 3 pistas ocultas y registra tu puntaje al ranking"),
        ]

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
                    for idx, (m_id, _, _) in enumerate(botones):
                        r = pg.Rect(90, 215 + idx * 105, self.ANCHO - 180, 86)
                        if r.collidepoint(mx, my):
                            return m_id

            self.screen.fill(self.BG)
            t1 = self.font_title.render("UIS Grand Prix Foundationals Models", True, self.ACCENT)
            self.screen.blit(t1, (90, 42))

            jugadas_n = len(cargar_jugadas())
            srv_txt = (
                f"Servidor OK ({self.salud_servidor.get('gpu_nombre', 'CPU')}, "
                f"VRAM libre: {self.salud_servidor.get('vram_libre_mb', 0)} MB)"
                if self.salud_servidor
                else f"Servidor sin respuesta en {self.bridge.server_url} (Pulsa H para reintentar)"
            )
            info_lines = [
                f"Alumno: {self.bridge.student_name} ({self.bridge.student_id}) | Estrategia: {cfg.ESTRATEGIA}",
                f"Jugadas guardadas: {jugadas_n} | Precision sensores: {cfg.CONFIG_DATOS['precision_sensores']} | Ventana H: {cfg.CONFIG_DATOS['ventana_historia']}",
                srv_txt,
            ]
            for i, line in enumerate(info_lines):
                c = self.OK if (i == 2 and self.salud_servidor) else (self.BAD if i == 2 else self.MUTED)
                self.screen.blit(self.font_body.render(line, True, c), (90, 86 + i * 26))

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

    def _modo_grabar_jugadas(self) -> None:
        pg = self.pygame
        idx_pista = 0
        clave = PISTAS_TRAIN[idx_pista]
        est = estado_inicial(clave)
        jugadas = cargar_jugadas()
        ventana = max(1, min(4, int(cfg.CONFIG_DATOS.get("ventana_historia", 3))))
        hist: List[int] = [0] * ventana
        frame = 0
        autopiloto = False

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        guardar_jugadas(jugadas)
                        self.estado_menu_msg = f"Jugadas guardadas en data/mis_jugadas.json ({len(jugadas)} muestras)."
                        return
                    if event.key in (pg.K_n, pg.K_TAB):
                        idx_pista = (idx_pista + 1) % len(PISTAS_TRAIN)
                        clave = PISTAS_TRAIN[idx_pista]
                        est = estado_inicial(clave)
                        hist = [0] * ventana
                    if event.key == pg.K_r:
                        est = estado_inicial(clave)
                        hist = [0] * ventana
                    if event.key == pg.K_p:
                        autopiloto = not autopiloto
                    if event.key == pg.K_c:
                        jugadas = []
                        guardar_jugadas(jugadas)

            sensores, impactos = calcular_sensores_rayos(est, clave)
            keys = pg.key.get_pressed()
            conduciendo = False

            if autopiloto:
                accion = piloto_guia_local(sensores, est.velocidad)
                conduciendo = True
            elif keys[pg.K_DOWN] or keys[pg.K_s] or keys[pg.K_SPACE]:
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
                accion = 0

            if conduciendo and frame % RECORD_EVERY_N_FRAMES == 0:
                jugadas.append({
                    "pista": clave,
                    "sensores": sensores,
                    "historial": hist[-ventana:],
                    "accion": accion,
                })

            if conduciendo or est.velocidad > 0.05:
                paso_simulacion(est, accion, clave)
                hist.append(accion)
            frame += 1

            self._dibujar_pista_y_carro(clave, est, sensores, impactos)
            lineas = [
                ("MODO [1]: GRABAR JUGADAS", self.ACCENT),
                (f"Pista: {PISTAS[clave]['nombre']}", self.TEXTO),
                (f"Muestras guardadas: {len(jugadas)}", self.OK),
                (f"Cada N frames: {RECORD_EVERY_N_FRAMES}", self.MUTED),
                (f"Velocidad: {est.velocidad:.2f} m/s", self.TEXTO),
                (f"Progreso: {calcular_progreso_pct(est, clave):.1f}%", self.TEXTO),
                (f"Vueltas: {est.vueltas} | CP: {est.checkpoints}", self.TEXTO),
                (f"Accion actual: {ACCIONES_DISCRETAS[accion][0]}", self.WARN),
                (f"Guia auto [P]: {'ON' if autopiloto else 'OFF'}", self.OK if autopiloto else self.MUTED),
                ("", self.TEXTO),
                ("Controles:", self.ACCENT),
                ("  Flechas / WASD : Conducir", self.MUTED),
                ("  Espacio / S    : Frenar", self.MUTED),
                ("  N / TAB        : Siguiente A", self.MUTED),
                ("  R              : Reiniciar", self.MUTED),
                ("  C              : Limpiar JSON", self.MUTED),
                ("  P              : Guia auto", self.MUTED),
                ("  ESC            : Guardar y salir", self.MUTED),
            ]
            self._dibujar_panel_lateral(lineas, sensores)
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
        ventana = max(1, min(4, int(cfg.CONFIG_DATOS.get("ventana_historia", 3))))
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
            self._dibujar_panel_lateral(lineas, sensores)
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
