"""TU ZONA DE TRABAJO (2 de 2): las piezas de PPO vistas en clase.

Todo lo que hay abajo funciona: es PPO estándar y es lo que usa `entrenar.py`.
Está aquí para que puedas leerlo, tocarlo y ver qué pasa. Ideas:
- lambda = 0 (solo la diferencia temporal) o lambda = 1 (retorno completo);
- quitar el recorte y observar cómo se desestabiliza el entrenamiento;
- no normalizar las ventajas;
- cambiar la pérdida del crítico.

La única condición es respetar las firmas de las funciones.
"""
import numpy as np
import torch


def calcular_ventajas(recompensas, valores, valores_siguientes, fin_episodio, gamma, lam):
    """Estimación generalizada de la ventaja (GAE).

    Con los símbolos de clase:

        delta_t = r_{t+1} + gamma * V(s_{t+1}) - V(s_t)          residuo de diferencia temporal
        A_t     = delta_t + gamma * lambda * A_{t+1}              suma ponderada de los residuos

    Los arreglos tienen forma (T, N): T pasos de N pistas en paralelo.
    - `valores_siguientes[t]` ya vale 0 donde el episodio terminó de verdad (meta):
      desde la meta no queda nada por ganar.
    - `fin_episodio[t]` es True donde acabó un episodio por cualquier motivo (meta o
      tiempo agotado): ahí se corta la recursión, porque el paso siguiente pertenece
      a otro episodio.
    """
    deltas = recompensas + gamma * valores_siguientes - valores
    ventajas = np.zeros_like(deltas)
    acumulado = np.zeros(deltas.shape[1:], dtype=float)
    for t in reversed(range(len(deltas))):
        acumulado = deltas[t] + gamma * lam * (1.0 - fin_episodio[t]) * acumulado
        ventajas[t] = acumulado
    return ventajas


def normalizar_ventajas(ventajas):
    """Ventajas con media 0 y desviación 1 dentro del lote.

    No cambia qué maniobras se empujan hacia arriba o hacia abajo, solo la escala
    del empujón; hace que la tasa de aprendizaje signifique lo mismo en todas las
    iteraciones.
    """
    return (ventajas - ventajas.mean()) / (ventajas.std() + 1e-8)


def perdida_actor(razon, ventajas, epsilon):
    """Objetivo recortado de PPO (con signo negativo, porque el optimizador minimiza).

        L^CLIP = mean( min( r_t * A_t,  clip(r_t, 1 - epsilon, 1 + epsilon) * A_t ) )

    `razon` = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t), un tensor por muestra.
    """
    sin_recorte = razon * ventajas
    con_recorte = torch.clamp(razon, 1.0 - epsilon, 1.0 + epsilon) * ventajas
    return -torch.min(sin_recorte, con_recorte).mean()


def perdida_critico(valores_predichos, retornos_objetivo):
    """Pérdida del crítico.

        L^VF = mean( ( V_phi(s_t) - objetivo_t )^2 ),   objetivo_t = A_t + V_old(s_t)
    """
    return ((valores_predichos - retornos_objetivo) ** 2).mean()
