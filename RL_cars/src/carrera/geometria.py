"""Utilidades geométricas del simulador.

Son las mismas del reto de algoritmos genéticos, más dos funciones para medir
el *progreso* del carro a lo largo de la línea central (la señal densa que
necesita una recompensa por paso).
"""
import math


def limitar(x, a, b):
    return a if x < a else b if x > b else x


def envolver_angulo(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def proyeccion_en_segmento(px, py, ax, ay, bx, by):
    x, y = bx - ax, by - ay
    den = x * x + y * y
    if den <= 1e-12:
        return ax, ay, 0.0
    t = limitar(((px - ax) * x + (py - ay) * y) / den, 0.0, 1.0)
    return ax + t * x, ay + t * y, t


def error_lateral_firmado(px, py, puntos):
    """Distancia del carro a la línea central, con signo según el lado."""
    mejor = float("inf")
    firmado = 0.0
    n = len(puntos)
    for i in range(n):
        ax, ay = puntos[i]
        bx, by = puntos[(i + 1) % n]
        qx, qy, _ = proyeccion_en_segmento(px, py, ax, ay, bx, by)
        dx, dy = px - qx, py - qy
        d = math.hypot(dx, dy)
        if d < mejor:
            cruz = (bx - ax) * (py - qy) - (by - ay) * (px - qx)
            mejor = d
            firmado = (1 if cruz >= 0 else -1) * d
    return firmado


def curvatura_local(puntos, indice):
    n = len(puntos)
    a = puntos[indice % n]
    b = puntos[(indice + 1) % n]
    c = puntos[(indice + 2) % n]
    a1 = math.atan2(b[1] - a[1], b[0] - a[0])
    a2 = math.atan2(c[1] - b[1], c[0] - b[0])
    return envolver_angulo(a2 - a1) / math.pi


def longitud_pista(puntos):
    """Perímetro de la línea central cerrada (metros)."""
    n = len(puntos)
    return sum(
        math.hypot(puntos[(i + 1) % n][0] - puntos[i][0], puntos[(i + 1) % n][1] - puntos[i][1])
        for i in range(n)
    )


def progreso_en_pista(px, py, puntos):
    """Posición (en metros) de la proyección del carro sobre la línea central.

    Vale 0 en el punto de salida y crece en el sentido de la carrera hasta el
    perímetro total, donde vuelve a 0.
    """
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


def avance(progreso_anterior, progreso_nuevo, longitud):
    """Metros avanzados en un paso, corrigiendo el cruce de la línea de salida."""
    delta = progreso_nuevo - progreso_anterior
    if delta > longitud / 2:
        delta -= longitud
    elif delta < -longitud / 2:
        delta += longitud
    return delta
