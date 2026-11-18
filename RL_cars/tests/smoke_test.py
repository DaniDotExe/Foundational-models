"""Prueba rápida de que todo el reto funciona. Ejecutar desde la raíz:

    python tests/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.carrera.simulador import (                                                     # noqa: E402
    NUM_ACCIONES, NUM_SENSORES, DT, PENALIZACION_CHOQUE, PENALIZACION_FUERA,
    ejecutar_prueba, estado_inicial, paso_simulacion,
)
from src.carrera.entorno import EntornoCarrera, EntornoVectorial                        # noqa: E402
from src.carrera.pistas import ORDEN_PISTAS, PISTAS, obstaculos_en                      # noqa: E402
from src.carrera.red import Actor, aplanar_actor, cargar_actor                          # noqa: E402
from src.carrera.politica import PoliticaNumpy, NUM_PESOS                               # noqa: E402
from src.carrera.limites import validar_hiperparametros                                 # noqa: E402
import mi_algoritmo as ppo                                                              # noqa: E402
import mi_recompensa                                                                    # noqa: E402
from mi_configuracion import CONFIG                                                     # noqa: E402
import entrenar                                                                         # noqa: E402

rng = np.random.default_rng(0)

# 1. el entorno responde y la recompensa del estudiante devuelve un número finito
entorno = EntornoCarrera("cantera", inicio_aleatorio=True, semilla=1)
obs = entorno.reset()
assert obs.shape == (NUM_SENSORES,)
for _ in range(50):
    obs, info, terminado, truncado = entorno.step(int(rng.integers(NUM_ACCIONES)))
    r = mi_recompensa.recompensa(info)
    assert np.isfinite(r), "la recompensa debe ser un número finito"
    if terminado or truncado:
        obs = entorno.reset()

# 2. el actor de PyTorch y la política NumPy son la misma función
actor = Actor()
vector = aplanar_actor(actor)
assert vector.shape == (NUM_PESOS,)
politica = PoliticaNumpy(vector)
sensores = rng.normal(size=NUM_SENSORES)
logits_torch = actor(torch.as_tensor(sensores, dtype=torch.float32)).detach().numpy()
assert np.allclose(logits_torch, politica.logits(sensores), atol=1e-5)
assert np.allclose(aplanar_actor(cargar_actor(vector)), vector)

# 3. una prueba oficial corre de principio a fin
resultado = ejecutar_prueba(politica, "ovalo", 1, 2.0)
assert resultado.tiempo > 0

# 4. GAE con lambda = 1 reproduce el retorno descontado menos el valor
T, N = 6, 1
recompensas = rng.normal(size=(T, N))
valores = rng.normal(size=(T, N))
valores_sig = np.vstack([valores[1:], np.zeros((1, N))])
fin = np.zeros((T, N))
fin[-1] = 1.0
gamma = 0.9
ventajas = ppo.calcular_ventajas(recompensas, valores, valores_sig, fin, gamma, 1.0)
retornos = np.zeros((T, N))
acumulado = 0.0
for t in reversed(range(T)):
    acumulado = recompensas[t] + gamma * acumulado
    retornos[t] = acumulado
assert np.allclose(ventajas, retornos - valores), "GAE con lambda=1 debe ser el retorno Monte Carlo menos V"

# 5. el vector de entornos y una configuración válida
vectorial = EntornoVectorial(ORDEN_PISTAS, 1, 1.0, True, 3)
obs = vectorial.reset()
salida = vectorial.step(rng.integers(NUM_ACCIONES, size=len(ORDEN_PISTAS)), mi_recompensa.recompensa)
assert salida[0].shape == (len(ORDEN_PISTAS), NUM_SENSORES)
validar_hiperparametros(CONFIG)

# 6. los obstáculos de Hora pico se mueven, y chocar o salirse suma tiempo al reloj de carrera
hora_pico = PISTAS["hora_pico"]
assert obstaculos_en(hora_pico, 0.0) != obstaculos_en(hora_pico, 1.0), "los obstáculos de Hora pico deben moverse"
estado = estado_inicial("hora_pico")
estado.tiempo = 1.0 - DT
ox, oy, _ = obstaculos_en(hora_pico, 1.0)[0]
estado.x, estado.y = ox, oy - 0.1                    # el obstáculo avanza hacia el carro detenido
for _ in range(15):
    paso_simulacion(estado, 6, "hora_pico")          # frenar: el obstáculo lo arrastra un rato
assert estado.colisiones_obstaculos == 1, "quedar pegado a un obstáculo es un solo choque"
assert abs(estado.tiempo_carrera - (estado.tiempo + PENALIZACION_CHOQUE)) < 1e-9

estado = estado_inicial("ovalo")
estado.x, estado.y = 20.0, 20.0
info = paso_simulacion(estado, 6, "ovalo")
assert info.fuera_pista and abs(info.penalizacion - PENALIZACION_FUERA * DT) < 1e-9

resultado = ejecutar_prueba(lambda sensores: 1, "ovalo", 1, 3.0)     # recto: se sale en la primera curva
assert resultado.tiempo_carrera >= 3.0 > resultado.tiempo, "el límite se mide con el reloj de carrera"

# 7. un entrenamiento mínimo completo, sin tocar la configuración del estudiante
minima = dict(CONFIG, pasos_totales=20_000, pasos_por_rollout=128, epocas=1, tam_minibatch=64, evaluar_cada=1)
with tempfile.TemporaryDirectory() as carpeta:
    mejor, puntaje, historial = entrenar.entrenar(minima, verbose=False, carpeta_salida=carpeta)
    assert mejor is not None and np.isfinite(puntaje)
    assert (Path(carpeta) / "historial.csv").exists()

print("Smoke test estudiantes OK")
