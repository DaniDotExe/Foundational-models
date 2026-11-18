"""El piloto entregable: un actor de tamaño fijo guardado como un vector de pesos.

Arquitectura fija (la misma para todos los estudiantes):

    12 sensores -> 32 (tanh) -> 32 (tanh) -> 7 logits -> softmax

Los pesos se aplanan en este orden: W1 (32x12), b1 (32), W2 (32x32), b2 (32),
W3 (7x32), b3 (7): 1703 números. El visualizador web reconstruye la red con la
misma receta, así que el .npy que entregas es exactamente lo que corre en el torneo.

Este módulo solo usa NumPy: sirve para verificar entregas sin PyTorch.
"""
import numpy as np

from .simulador import NUM_SENSORES, NUM_ACCIONES

TAMANO_OCULTO = 32
FORMAS_ACTOR = [
    (TAMANO_OCULTO, NUM_SENSORES), (TAMANO_OCULTO,),
    (TAMANO_OCULTO, TAMANO_OCULTO), (TAMANO_OCULTO,),
    (NUM_ACCIONES, TAMANO_OCULTO), (NUM_ACCIONES,),
]
NUM_PESOS = int(sum(int(np.prod(forma)) for forma in FORMAS_ACTOR))  # 1703
LIMITE_PESO = 50.0


def validar_pesos(vector):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.shape != (NUM_PESOS,):
        raise ValueError(f"El piloto debe tener exactamente {NUM_PESOS} pesos; llegaron {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("El vector de pesos contiene NaN o infinito.")
    if np.any(np.abs(vector) > LIMITE_PESO):
        raise ValueError(f"Todos los pesos deben estar en [-{LIMITE_PESO:g}, {LIMITE_PESO:g}].")
    return vector


def desaplanar(vector):
    """Convierte el vector de 1703 pesos en la lista [W1, b1, W2, b2, W3, b3]."""
    vector = validar_pesos(vector)
    capas = []
    inicio = 0
    for forma in FORMAS_ACTOR:
        tamano = int(np.prod(forma))
        capas.append(vector[inicio:inicio + tamano].reshape(forma))
        inicio += tamano
    return capas


class PoliticaNumpy:
    """Actor determinista (argmax) construido solo con NumPy."""

    def __init__(self, vector):
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = desaplanar(vector)

    def logits(self, sensores):
        h = np.tanh(self.w1 @ np.asarray(sensores, dtype=float) + self.b1)
        h = np.tanh(self.w2 @ h + self.b2)
        return self.w3 @ h + self.b3

    def probabilidades(self, sensores):
        z = self.logits(sensores)
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def __call__(self, sensores):
        return int(np.argmax(self.logits(sensores)))
