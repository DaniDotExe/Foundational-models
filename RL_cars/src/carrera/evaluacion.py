"""La prueba oficial: cómo se compara a los pilotos.

La misma regla se usa para dos cosas:
- durante el entrenamiento, para decidir qué actor se guarda como `mejor_actor.npy`;
- en el torneo, para ordenar las entregas.
"""
from .pistas import ORDEN_PISTAS, PISTAS
from .simulador import ejecutar_prueba

VUELTAS_PRUEBA = 1
TIEMPO_PRUEBA = 24.0
VUELTAS_FINAL = 5
TIEMPO_FINAL = 150.0

PESO_VUELTA = 1000.0
PESO_CHECKPOINT = 50.0


def puntaje_pista(resultado):
    """Primero completar, luego rapidez con limpieza (más alto es mejor).

    `tiempo_carrera` ya trae los castigos: cada segundo fuera de la pista suma
    PENALIZACION_FUERA s y cada choque PENALIZACION_CHOQUE s (ver simulador.py).
    """
    return (
        PESO_VUELTA * resultado.vueltas
        + PESO_CHECKPOINT * resultado.checkpoints
        - resultado.tiempo_carrera
    )


def puntaje_oficial(resultados):
    return float(sum(puntaje_pista(r) for r in resultados))


def evaluar_politica(politica, vueltas=VUELTAS_PRUEBA, tiempo_limite=TIEMPO_PRUEBA):
    resultados = [ejecutar_prueba(politica, pista, vueltas, tiempo_limite) for pista in ORDEN_PISTAS]
    return resultados, puntaje_oficial(resultados)


def imprimir_tabla(resultados):
    for r in resultados:
        print(
            f"{PISTAS[r.pista]['nombre']:20s} | vueltas={r.vueltas} | CP={r.checkpoints:3d} | "
            f"tiempo={r.tiempo:6.2f}s | fuera={r.tiempo_fuera:5.2f}s | choques={r.colisiones_obstaculos:2d} | "
            f"castigo=+{r.tiempo_penalizacion:5.2f}s | reloj={r.tiempo_carrera:6.2f}s | puntos={puntaje_pista(r):8.1f}"
        )
    print(f"{'PUNTAJE OFICIAL':20s} | {puntaje_oficial(resultados):.1f}")
