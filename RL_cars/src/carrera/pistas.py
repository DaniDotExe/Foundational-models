"""Las pistas oficiales del reto: las cinco del reto genético y una sexta con obstáculos en movimiento.

Obstáculos:
- `obstaculos`: fijos, cada uno (x, y, radio).
- `obstaculos_moviles`: cada uno (ax, ay, bx, by, radio, periodo, fase). Va y viene
  entre A y B con un vaivén suave: sale de A, llega a B a medio periodo y vuelve.
  `fase` (0..1) es la fracción de periodo que ya lleva recorrida cuando empieza el
  episodio. Su posición depende solo del tiempo del episodio, así que la prueba
  oficial sigue siendo determinista.
"""
import math

from .geometria import longitud_pista

PISTAS = {
    "ovalo": {
        "nombre": "1 - Óvalo", "dificultad": 1, "ancho_medio": 2.35,
        "puntos": [(-8.0, 0.0), (-6.5, -4.0), (-2.5, -5.6), (2.5, -5.6), (6.5, -4.0),
                   (8.0, 0.0), (6.5, 4.0), (2.5, 5.6), (-2.5, 5.6), (-6.5, 4.0)],
        "obstaculos": [],
        "obstaculos_moviles": [],
    },
    "curvas_largas": {
        "nombre": "2 - Curvas largas", "dificultad": 2, "ancho_medio": 2.25,
        "puntos": [(-8.3, -0.6), (-6.3, -4.4), (-1.5, -5.5), (3.6, -4.8), (7.5, -2.0),
                   (8.1, 2.1), (5.0, 5.3), (0.2, 5.6), (-3.9, 4.4), (-7.5, 2.3)],
        "obstaculos": [],
        "obstaculos_moviles": [],
    },
    "chicane": {
        "nombre": "3 - Chicane", "dificultad": 3, "ancho_medio": 2.10,
        "puntos": [(-8.2, 0.0), (-6.4, -4.4), (-2.0, -5.5), (0.4, -2.0), (3.0, -5.1), (7.0, -3.5),
                   (8.2, 0.5), (5.8, 4.8), (1.8, 5.5), (-0.7, 2.0), (-3.7, 5.0), (-7.1, 3.2)],
        "obstaculos": [],
        "obstaculos_moviles": [],
    },
    "serpiente": {
        "nombre": "4 - Serpiente", "dificultad": 4, "ancho_medio": 1.95,
        "puntos": [(-8.4, 1.2), (-7.0, -4.6), (-3.1, -5.5), (-0.3, -2.2), (2.4, -5.3), (6.6, -4.0),
                   (8.3, -0.8), (6.0, 1.3), (8.0, 4.3), (3.7, 5.7), (0.7, 2.5), (-2.2, 5.4), (-6.5, 4.4)],
        "obstaculos": [],
        "obstaculos_moviles": [],
    },
    "cantera": {
        "nombre": "5 - La cantera", "dificultad": 5, "ancho_medio": 1.90,
        "puntos": [(-8.5, 0.8), (-7.1, -4.5), (-3.0, -5.6), (1.3, -5.2), (4.7, -3.7), (7.9, -1.3),
                   (7.3, 2.0), (4.6, 4.9), (0.8, 5.6), (-1.2, 2.5), (-4.4, 5.0), (-7.8, 3.5)],
        "obstaculos": [(-4.4, -4.55, 0.62), (2.8, -4.35, 0.68), (6.75, 0.55, 0.72),
                       (2.85, 4.85, 0.62), (-2.55, 3.70, 0.65), (-7.00, 2.20, 0.58)],
        "obstaculos_moviles": [],
    },
    "hora_pico": {
        # Trazado amable; lo difícil es el tráfico que cruza la pista de borde a borde.
        "nombre": "6 - Hora pico", "dificultad": 6, "ancho_medio": 2.10,
        "puntos": [(-8.4, 0.0), (-7.3, -4.0), (-3.8, -5.6), (0.0, -5.6), (3.8, -5.6), (7.3, -4.0),
                   (8.4, 0.0), (7.3, 4.0), (3.8, 5.6), (0.0, 4.9), (-3.8, 5.6), (-7.3, 4.0)],
        "obstaculos": [],
        "obstaculos_moviles": [(-1.90, -7.70, -1.90, -3.50, 0.65, 4.0, 0.00),
                               (1.90, -3.50, 1.90, -7.70, 0.60, 5.0, 0.00),
                               (9.87, 2.56, 5.83, 1.44, 0.70, 4.5, 0.25),
                               (2.28, 3.18, 1.52, 7.32, 0.65, 5.5, 0.60),
                               (-5.83, 1.44, -9.87, 2.56, 0.60, 3.6, 0.40)],
    },
}
ORDEN_PISTAS = ["ovalo", "curvas_largas", "chicane", "serpiente", "cantera", "hora_pico"]

for _pista in PISTAS.values():
    _pista["longitud"] = longitud_pista(_pista["puntos"])


def obstaculos_en(pista, tiempo):
    """Todos los obstáculos de `pista` como (x, y, radio) en el instante `tiempo` del episodio."""
    obstaculos = list(pista["obstaculos"])
    for ax, ay, bx, by, radio, periodo, fase in pista["obstaculos_moviles"]:
        u = 0.5 - 0.5 * math.cos(2 * math.pi * (tiempo / periodo + fase))
        obstaculos.append((ax + u * (bx - ax), ay + u * (by - ay), radio))
    return obstaculos
