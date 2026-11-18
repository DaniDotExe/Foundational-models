"""Hiperparámetros de TU experimento de PPO.

Puedes cambiar estos valores, pero el entrenador los validará antes de empezar.
Igual que en el reto genético, existe un presupuesto máximo (pasos de
interacción con el entorno) para que gane el diseño y no la fuerza bruta.
"""

CONFIG = {
    # Presupuesto: cuántos pasos de 0.06 s puede vivir el carro en total (máx. 600 000).
    "pasos_totales": 300_000,

    # Recolección: pasos por pista antes de cada actualización.
    # Las 6 pistas corren en paralelo, así que cada lote tiene pasos_por_rollout x 6 muestras.
    "pasos_por_rollout": 400,

    # Optimización: cuántas veces se recorre el mismo lote (K) y en trozos de qué tamaño.
    "epocas": 8,
    "tam_minibatch": 250,
    "tasa_aprendizaje": 1e-3,

    # Retorno y ventaja: descuento gamma y el lambda de GAE.
    "gamma": 0.99,
    "lam": 0.95,

    # PPO: ancho de la banda de confianza (epsilon) y peso de la pérdida del crítico (c1).
    "epsilon_clip": 0.2,
    "coef_valor": 0.5,

    # Bono de exploración opcional. 0.0 = la fórmula exacta vista en clase.
    "coef_entropia": 0.0,

    # Empezar los episodios de entrenamiento en checkpoints al azar (la prueba oficial siempre sale desde la meta).
    "inicio_aleatorio": True,

    # Cada cuántas actualizaciones se corre la prueba oficial para guardar el mejor actor.
    "evaluar_cada": 5,

    "semilla": 7,
}
