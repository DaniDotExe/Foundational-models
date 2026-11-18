"""Física del carro, sensores y maniobras discretas.

La física es exactamente la del reto de algoritmos genéticos: el mismo carro,
los mismos 12 sensores y las mismas cinco pistas (más una sexta con obstáculos
en movimiento). Lo único nuevo es que el piloto ya no entrega tres números
continuos: en cada paso elige una de las NUM_ACCIONES maniobras de la lista ACCIONES.

Castigos de tiempo: chocar y salirse de la pista suman segundos al reloj de
carrera (`tiempo_carrera`). Ese reloj es el que agota el límite de la prueba y el
que resta en el puntaje, así que un piloto que choca tiene menos tiempo para
completar sus vueltas.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np

from .pistas import PISTAS, obstaculos_en
from .geometria import (
    limitar, envolver_angulo, error_lateral_firmado, curvatura_local,
    progreso_en_pista, avance,
)

NUM_SENSORES = 12
DT = 0.06
VELOCIDAD_MAX = 7.6
ACELERACION = 5.2
FRENO = 8.8
ARRASTRE = 0.20
GIRO_MAX = 2.20
RADIO_CHECKPOINT = 1.45
RADIO_CARRO = 0.28
MARGEN_CONTACTO = 0.05       # m: mientras siga así de pegado al obstáculo, es el mismo choque

PENALIZACION_FUERA = 1.0     # segundos extra en el reloj por cada segundo fuera de la pista
PENALIZACION_CHOQUE = 2.0    # segundos extra en el reloj por cada choque con un obstáculo

# Maniobras del piloto: (nombre, acelerador, volante, freno).
# El volante vale -1 (izquierda), 0 (recto) o +1 (derecha) tal como lo dibuja el visualizador.
ACCIONES = [
    ("acelerar + izquierda", 1.0, -1.0, 0.0),
    ("acelerar + recto", 1.0, 0.0, 0.0),
    ("acelerar + derecha", 1.0, 1.0, 0.0),
    ("rodar + izquierda", 0.0, -1.0, 0.0),
    ("rodar + recto", 0.0, 0.0, 0.0),
    ("rodar + derecha", 0.0, 1.0, 0.0),
    ("frenar", 0.0, 0.0, 1.0),
]
NUM_ACCIONES = len(ACCIONES)
NOMBRES_ACCIONES = [accion[0] for accion in ACCIONES]


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
    distancia_recorrida: float = 0.0
    velocidad_maxima: float = 0.0
    suma_error_lateral: float = 0.0
    pasos: int = 0
    suma_variacion_giro: float = 0.0
    ultimo_giro: float = 0.0
    progreso: float = 0.0
    tiempo_penalizacion: float = 0.0
    contactos: tuple = ()         # índices de los obstáculos que está tocando

    @property
    def tiempo_carrera(self):
        """El reloj oficial: tiempo simulado más los castigos por choques y por salirse."""
        return self.tiempo + self.tiempo_penalizacion


@dataclass
class InfoPaso:
    """Todo lo que pasó en un paso de 0.06 s. Es la materia prima de tu recompensa."""
    pista: str
    tiempo: float                 # segundos simulados en el episodio (sin castigos)
    tiempo_carrera: float         # reloj oficial: tiempo + castigos acumulados
    dt: float                     # duración del paso (0.06 s)
    accion: int                   # índice de la maniobra elegida (0..6)
    nombre_accion: str
    velocidad: float              # m/s después del paso (0 .. 7.6)
    velocidad_maxima: float       # constante: 7.6 m/s
    progreso: float               # metros avanzados sobre la línea central (negativo si retrocede)
    distancia_al_siguiente: float # metros al próximo checkpoint
    error_lateral: float          # metros al centro de la pista (siempre >= 0)
    ancho_medio: float            # medio ancho de la pista: fuera si error_lateral > ancho_medio
    fuera_pista: bool             # el carro está fuera de la pista en este paso
    colision: bool                # chocó con un obstáculo en este paso
    penalizacion: float           # segundos de castigo que este paso sumó al reloj
    checkpoint_alcanzado: bool    # cruzó el siguiente checkpoint en este paso
    vuelta_completada: bool       # cruzó la línea de meta en este paso
    cambio_giro: float            # |volante actual - volante anterior| (0, 1 o 2)
    termino: bool = False         # el episodio terminó porque se completó el objetivo
    tiempo_agotado: bool = False  # el episodio terminó por límite de tiempo

    def como_diccionario(self):
        return asdict(self)


@dataclass
class ResultadoPrueba:
    """Resumen de una prueba completa (igual que en el reto genético, más los castigos de tiempo)."""
    pista: str
    tiempo: float
    tiempo_penalizacion: float
    tiempo_carrera: float
    vueltas: int
    checkpoints: int
    checkpoint_actual: int
    tiempo_fuera: float
    colisiones_obstaculos: int
    distancia_recorrida: float
    velocidad_maxima: float
    termino_objetivo: bool
    distancia_al_siguiente: float
    velocidad_promedio: float
    error_lateral_medio: float
    variacion_direccion: float

    def como_diccionario(self):
        return asdict(self)


def estado_inicial(nombre_pista, indice_inicio=0):
    """Carro detenido sobre el checkpoint `indice_inicio`, mirando al siguiente."""
    puntos = PISTAS[nombre_pista]["puntos"]
    n = len(puntos)
    x0, y0 = puntos[indice_inicio % n]
    x1, y1 = puntos[(indice_inicio + 1) % n]
    estado = EstadoCarro(x0, y0, math.atan2(y1 - y0, x1 - x0))
    estado.siguiente_checkpoint = (indice_inicio + 1) % n
    estado.progreso = progreso_en_pista(x0, y0, puntos)
    return estado


def _sensor_obstaculo(estado, pista):
    obstaculos = obstaculos_en(pista, estado.tiempo)
    if not obstaculos:
        return 0.0, 0.0, 0.0
    mejor = None
    mejor_distancia = 1e9
    for ox, oy, r in obstaculos:
        d = math.hypot(ox - estado.x, oy - estado.y) - r
        if d < mejor_distancia:
            mejor_distancia = d
            mejor = (ox, oy, r)
    ox, oy, _ = mejor
    relativo = envolver_angulo(math.atan2(oy - estado.y, ox - estado.x) - estado.angulo)
    return limitar((6 - max(0.0, mejor_distancia)) / 6, 0.0, 1.0), math.cos(relativo), math.sin(relativo)


def obtener_sensores(estado, nombre_pista):
    """Los 12 números que ve el piloto (idénticos al reto genético)."""
    pista = PISTAS[nombre_pista]
    puntos = pista["puntos"]
    tx, ty = puntos[estado.siguiente_checkpoint]
    dx, dy = tx - estado.x, ty - estado.y
    distancia = math.hypot(dx, dy)
    relativo = envolver_angulo(math.atan2(dy, dx) - estado.angulo)
    error = error_lateral_firmado(estado.x, estado.y, puntos)
    ancho = pista["ancho_medio"]
    obstaculo = _sensor_obstaculo(estado, pista)
    return np.array([
        1.0,
        math.cos(relativo),
        math.sin(relativo),
        limitar(distancia / 12, 0.0, 1.0),
        limitar(error / ancho, -2.0, 2.0),
        limitar(abs(error) / ancho, 0.0, 2.0),
        limitar(estado.velocidad / VELOCIDAD_MAX, 0.0, 1.5),
        curvatura_local(puntos, estado.siguiente_checkpoint),
        curvatura_local(puntos, estado.siguiente_checkpoint + 1),
        obstaculo[0],
        obstaculo[1],
        obstaculo[2],
    ], dtype=float)


def _resolver_obstaculos(estado, obstaculos):
    """Empuja el carro fuera de los obstáculos que tocó. Devuelve cuántos choques nuevos hubo.

    Un choque se cuenta una sola vez: mientras el carro siga pegado al mismo obstáculo
    (empujando contra él, o arrastrado por uno que se mueve) no vuelve a contar.
    """
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


def paso_simulacion(estado, accion, nombre_pista):
    """Aplica una maniobra durante DT segundos y devuelve lo que pasó (InfoPaso)."""
    pista = PISTAS[nombre_pista]
    puntos = pista["puntos"]
    nombre, acel, giro, freno = ACCIONES[int(accion)]
    e = estado

    cambio_giro = abs(giro - e.ultimo_giro)
    e.suma_variacion_giro += cambio_giro
    e.ultimo_giro = giro

    e.velocidad = limitar(
        e.velocidad + (acel * ACELERACION - freno * FRENO - ARRASTRE * e.velocidad) * DT,
        0.0, VELOCIDAD_MAX,
    )
    e.angulo = envolver_angulo(e.angulo + giro * GIRO_MAX * (0.28 + 0.72 * e.velocidad / VELOCIDAD_MAX) * DT)
    x0, y0 = e.x, e.y
    e.x += math.cos(e.angulo) * e.velocidad * DT
    e.y += math.sin(e.angulo) * e.velocidad * DT
    e.distancia_recorrida += math.hypot(e.x - x0, e.y - y0)
    e.velocidad_maxima = max(e.velocidad_maxima, e.velocidad)
    e.tiempo += DT

    error_lateral = abs(error_lateral_firmado(e.x, e.y, puntos))
    e.suma_error_lateral += error_lateral
    e.pasos += 1
    fuera = error_lateral > pista["ancho_medio"]
    penalizacion = 0.0
    if fuera:
        e.tiempo_fuera += DT
        e.velocidad *= 0.76
        penalizacion += PENALIZACION_FUERA * DT
    choques = _resolver_obstaculos(e, obstaculos_en(pista, e.tiempo))
    penalizacion += PENALIZACION_CHOQUE * choques
    e.tiempo_penalizacion += penalizacion

    progreso_nuevo = progreso_en_pista(e.x, e.y, puntos)
    metros = avance(e.progreso, progreso_nuevo, pista["longitud"])
    e.progreso = progreso_nuevo

    checkpoint = False
    vuelta = False
    tx, ty = puntos[e.siguiente_checkpoint]
    if math.hypot(e.x - tx, e.y - ty) <= RADIO_CHECKPOINT:
        e.siguiente_checkpoint = (e.siguiente_checkpoint + 1) % len(puntos)
        e.checkpoints += 1
        checkpoint = True
        if e.siguiente_checkpoint == 1:
            e.vueltas += 1
            vuelta = True
    tx, ty = puntos[e.siguiente_checkpoint]

    return InfoPaso(
        pista=nombre_pista,
        tiempo=e.tiempo,
        tiempo_carrera=e.tiempo_carrera,
        dt=DT,
        accion=int(accion),
        nombre_accion=nombre,
        velocidad=e.velocidad,
        velocidad_maxima=VELOCIDAD_MAX,
        progreso=metros,
        distancia_al_siguiente=math.hypot(e.x - tx, e.y - ty),
        error_lateral=error_lateral,
        ancho_medio=pista["ancho_medio"],
        fuera_pista=fuera,
        colision=choques > 0,
        penalizacion=penalizacion,
        checkpoint_alcanzado=checkpoint,
        vuelta_completada=vuelta,
        cambio_giro=cambio_giro,
    )


def ejecutar_prueba(politica, nombre_pista, vueltas_objetivo=1, tiempo_limite=24.0, indice_inicio=0):
    """Corre una prueba completa con una política determinista.

    `politica` es cualquier callable que reciba los 12 sensores y devuelva el
    índice de la maniobra (por ejemplo, una PoliticaNumpy cargada de un .npy).
    El límite de tiempo se mide con el reloj de carrera (castigos incluidos).
    """
    estado = estado_inicial(nombre_pista, indice_inicio)
    while estado.tiempo_carrera < tiempo_limite and estado.vueltas < vueltas_objetivo:
        accion = politica(obtener_sensores(estado, nombre_pista))
        paso_simulacion(estado, accion, nombre_pista)

    puntos = PISTAS[nombre_pista]["puntos"]
    tx, ty = puntos[estado.siguiente_checkpoint]
    velocidad_promedio = estado.distancia_recorrida / estado.tiempo if estado.tiempo > 0 else 0.0
    error_lateral_medio = estado.suma_error_lateral / estado.pasos if estado.pasos else 0.0
    variacion_direccion = estado.suma_variacion_giro / estado.pasos if estado.pasos else 0.0
    return ResultadoPrueba(
        pista=nombre_pista,
        tiempo=estado.tiempo,
        tiempo_penalizacion=estado.tiempo_penalizacion,
        tiempo_carrera=estado.tiempo_carrera,
        vueltas=estado.vueltas,
        checkpoints=estado.checkpoints,
        checkpoint_actual=estado.siguiente_checkpoint,
        tiempo_fuera=estado.tiempo_fuera,
        colisiones_obstaculos=estado.colisiones_obstaculos,
        distancia_recorrida=estado.distancia_recorrida,
        velocidad_maxima=estado.velocidad_maxima,
        termino_objetivo=estado.vueltas >= vueltas_objetivo,
        distancia_al_siguiente=math.hypot(estado.x - tx, estado.y - ty),
        velocidad_promedio=velocidad_promedio,
        error_lateral_medio=error_lateral_medio,
        variacion_direccion=variacion_direccion,
    )
