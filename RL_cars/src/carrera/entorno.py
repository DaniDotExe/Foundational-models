"""El simulador visto como un entorno de aprendizaje por refuerzo.

`EntornoCarrera` sigue la interfaz clásica reset()/step(accion) para una pista.
`EntornoVectorial` corre todas las pistas en paralelo y aplica TU función de
recompensa a lo que pasa en cada paso.
"""
import math

import numpy as np

from .pistas import PISTAS
from .simulador import estado_inicial, obtener_sensores, paso_simulacion


class EntornoCarrera:
    def __init__(self, nombre_pista, vueltas_objetivo=1, tiempo_limite=24.0,
                 inicio_aleatorio=False, semilla=None):
        self.nombre_pista = nombre_pista
        self.vueltas_objetivo = vueltas_objetivo
        self.tiempo_limite = tiempo_limite
        self.inicio_aleatorio = inicio_aleatorio
        self.rng = np.random.default_rng(semilla)
        self.estado = None

    def reset(self):
        n = len(PISTAS[self.nombre_pista]["puntos"])
        inicio = int(self.rng.integers(n)) if self.inicio_aleatorio else 0
        self.estado = estado_inicial(self.nombre_pista, inicio)
        return obtener_sensores(self.estado, self.nombre_pista)

    def step(self, accion):
        """Devuelve (sensores, info, terminado, truncado).

        terminado: se cumplió el objetivo (meta). El estado siguiente no tiene futuro: V = 0.
        truncado:  se acabó el tiempo. El episodio se corta, pero el estado siguiente sí tiene valor.
                   Cuenta el reloj de carrera: los castigos por choques y por salirse lo adelantan.
        """
        info = paso_simulacion(self.estado, int(accion), self.nombre_pista)
        terminado = self.estado.vueltas >= self.vueltas_objetivo
        truncado = (not terminado) and self.estado.tiempo_carrera >= self.tiempo_limite - 1e-9
        info.termino = terminado
        info.tiempo_agotado = truncado
        return obtener_sensores(self.estado, self.nombre_pista), info, terminado, truncado


class EntornoVectorial:
    """Varias pistas en paralelo, con reinicio automático al terminar cada episodio."""

    def __init__(self, nombres_pistas, vueltas_objetivo=1, tiempo_limite=24.0,
                 inicio_aleatorio=False, semilla=0):
        self.entornos = [
            EntornoCarrera(nombre, vueltas_objetivo, tiempo_limite, inicio_aleatorio, semilla + i)
            for i, nombre in enumerate(nombres_pistas)
        ]
        self.retornos = np.zeros(len(self.entornos))
        self.longitudes = np.zeros(len(self.entornos), dtype=int)

    def __len__(self):
        return len(self.entornos)

    def reset(self):
        self.retornos[:] = 0.0
        self.longitudes[:] = 0
        return np.stack([entorno.reset() for entorno in self.entornos])

    def step(self, acciones, recompensa):
        """Un paso en todas las pistas.

        Devuelve:
        - observaciones siguientes (N, 12), ya reiniciadas donde el episodio acabó;
        - recompensas (N,) calculadas con TU función;
        - fin_episodio (N,) True donde el episodio acabó por cualquier motivo;
        - terminal (N,) True donde acabó por cumplir el objetivo (el valor futuro es 0);
        - observaciones para bootstrap (N, 12): la última observación real del episodio,
          o la observación siguiente si el episodio continúa;
        - resúmenes: una lista con las estadísticas de los episodios que acabaron en este paso.
        """
        n = len(self.entornos)
        observaciones = np.zeros((n, 12))
        bootstrap = np.zeros((n, 12))
        recompensas = np.zeros(n)
        fin_episodio = np.zeros(n, dtype=bool)
        terminal = np.zeros(n, dtype=bool)
        resumenes = []

        for i, (entorno, accion) in enumerate(zip(self.entornos, acciones)):
            obs, info, terminado, truncado = entorno.step(accion)
            r = float(recompensa(info))
            if not math.isfinite(r):
                raise ValueError(f"Tu recompensa devolvió {r!r} en la pista {info.pista}; debe ser un número finito.")
            recompensas[i] = r
            self.retornos[i] += r
            self.longitudes[i] += 1
            bootstrap[i] = obs
            if terminado or truncado:
                estado = entorno.estado
                resumenes.append({
                    "pista": entorno.nombre_pista,
                    "retorno": float(self.retornos[i]),
                    "pasos": int(self.longitudes[i]),
                    "vueltas": estado.vueltas,
                    "checkpoints": estado.checkpoints,
                    "tiempo": estado.tiempo,
                    "tiempo_fuera": estado.tiempo_fuera,
                    "colisiones": estado.colisiones_obstaculos,
                    "tiempo_penalizacion": estado.tiempo_penalizacion,
                    "termino": terminado,
                })
                self.retornos[i] = 0.0
                self.longitudes[i] = 0
                fin_episodio[i] = True
                terminal[i] = terminado
                obs = entorno.reset()
            observaciones[i] = obs

        return observaciones, recompensas, fin_episodio, terminal, bootstrap, resumenes
