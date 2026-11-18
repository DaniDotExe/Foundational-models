"""TU ZONA DE TRABAJO (1 de 2): la recompensa.

En el reto genético diseñaste una *fitness*: una nota para toda la carrera,
calculada al final. Aquí diseñas una *recompensa*: un número r_{t+1} que el
entorno entrega después de CADA paso de 0.06 s.

El entrenador (PPO) hace el resto: estima cuánto mejor de lo esperado fue cada
maniobra (la ventaja) y empuja hacia arriba las que resultaron mejores y hacia
abajo las que resultaron peores. Tú decides qué significa "mejor".
"""
from src.carrera.simulador import InfoPaso, VELOCIDAD_MAX


def recompensa(paso: InfoPaso) -> float:
    """RECOMPENSA INTENCIONALMENTE MALA.

    Premia estar cerca del siguiente checkpoint: cuanto menor la distancia,
    menos negativa la recompensa. Suena razonable... pero piensa qué ocurre en
    el instante en que el carro cruza el checkpoint: el "siguiente" pasa a ser
    otro, mucho más lejano, y la recompensa cae de golpe. El agente maximiza
    exactamente lo que escribes, no lo que querías decir. Es el equivalente de
    la fitness "mala" del reto genético.

    `paso` es un InfoPaso con todo lo que ocurrió en este instante:

    Progreso:
    - paso.progreso              metros avanzados sobre la línea central (negativo si retrocede)
    - paso.checkpoint_alcanzado  True el paso en que cruza un checkpoint
    - paso.vuelta_completada     True el paso en que cruza la meta
    - paso.distancia_al_siguiente

    Rapidez:
    - paso.velocidad             m/s (0 .. 7.6)
    - paso.dt                    0.06 s: cobrar un poco por cada paso invita a terminar pronto

    Calidad de trayectoria:
    - paso.error_lateral         metros al centro; fuera si es mayor que paso.ancho_medio
    - paso.fuera_pista           True mientras está fuera

    Seguridad:
    - paso.colision              True el paso en que golpea un obstáculo (fijo en La cantera,
                                 en movimiento en Hora pico); rozarlo sin soltarse cuenta una vez

    Castigos de tiempo (los mismos que usa la prueba oficial):
    - paso.penalizacion          segundos que este paso sumó al reloj: 2 s por choque y
                                 0.06 s extra por cada paso fuera de la pista
    - paso.tiempo_carrera        el reloj oficial: tiempo + castigos. Es el que agota los 24 s,
                                 así que chocar o salirse deja menos tiempo para la vuelta

    Suavidad:
    - paso.cambio_giro           cuánto cambió el volante respecto al paso anterior (0, 1 o 2)

    Contexto:
    - paso.accion, paso.nombre_accion, paso.pista, paso.tiempo (sin castigos),
      paso.termino (objetivo cumplido), paso.tiempo_agotado

    Pistas para pensar: ¿qué comportamiento premia cada término? ¿qué pasa si
    todo es positivo? ¿y si un término domina a los demás? Recuerda que el
    agente maximiza la SUMA de recompensas del episodio, no una sola.
    """
    return -0.1 * paso.distancia_al_siguiente
