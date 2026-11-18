"""Actor y crítico en PyTorch (la arquitectura fija descrita en politica.py)."""
import math

import numpy as np
import torch
import torch.nn as nn

from .simulador import NUM_SENSORES, NUM_ACCIONES
from .politica import TAMANO_OCULTO, FORMAS_ACTOR, NUM_PESOS, validar_pesos


def _capa(entrada, salida, ganancia):
    capa = nn.Linear(entrada, salida)
    nn.init.orthogonal_(capa.weight, gain=ganancia)
    nn.init.zeros_(capa.bias)
    return capa


class Actor(nn.Module):
    """pi_theta(a | s): 12 sensores -> probabilidades sobre las 7 maniobras."""

    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            _capa(NUM_SENSORES, TAMANO_OCULTO, math.sqrt(2)), nn.Tanh(),
            _capa(TAMANO_OCULTO, TAMANO_OCULTO, math.sqrt(2)), nn.Tanh(),
            _capa(TAMANO_OCULTO, NUM_ACCIONES, 0.01),  # empieza casi uniforme: explora
        )

    def forward(self, observaciones):
        return self.red(observaciones)  # logits

    def distribucion(self, observaciones):
        return torch.distributions.Categorical(logits=self.forward(observaciones))


class Critico(nn.Module):
    """V_phi(s): 12 sensores -> un número, el retorno que aún se espera desde s."""

    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(
            _capa(NUM_SENSORES, TAMANO_OCULTO, math.sqrt(2)), nn.Tanh(),
            _capa(TAMANO_OCULTO, TAMANO_OCULTO, math.sqrt(2)), nn.Tanh(),
            _capa(TAMANO_OCULTO, 1, 1.0),
        )

    def forward(self, observaciones):
        return self.red(observaciones).squeeze(-1)


def aplanar_actor(actor):
    """Vector de NUM_PESOS números en el orden W1, b1, W2, b2, W3, b3."""
    partes = [parametro.detach().cpu().numpy().astype(np.float64).ravel() for parametro in actor.red.parameters()]
    vector = np.concatenate(partes)
    assert vector.shape == (NUM_PESOS,), vector.shape
    return vector


def cargar_actor(vector):
    """Reconstruye un Actor de PyTorch a partir de un vector de pesos."""
    vector = validar_pesos(vector)
    actor = Actor()
    inicio = 0
    with torch.no_grad():
        for parametro, forma in zip(actor.red.parameters(), FORMAS_ACTOR):
            tamano = int(np.prod(forma))
            parametro.copy_(torch.as_tensor(vector[inicio:inicio + tamano].reshape(forma), dtype=torch.float32))
            inicio += tamano
    return actor
