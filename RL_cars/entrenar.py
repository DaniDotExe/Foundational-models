#!/usr/bin/env python3
"""Entrenador oficial del reto (PPO).

No necesitas editar este archivo para competir. Concentra tu trabajo en:

- mi_recompensa.py     (qué significa conducir bien, paso a paso)
- mi_configuracion.py  (los hiperparámetros)
- mi_algoritmo.py      (las piezas de PPO, si quieres experimentar)

El ciclo es el de clase:
    1. recolectar trayectorias con la política vieja (las 6 pistas en paralelo);
    2. calcular los retornos objetivo y las ventajas (GAE);
    3. K épocas de pasos de gradiente sobre L = L^CLIP - c1 * L^VF;
    4. theta_old <- theta y repetir.

Cada `evaluar_cada` actualizaciones se corre la prueba oficial, que es la misma
final del torneo (5 vueltas o 150 s por pista), y se guarda el mejor actor. Se
elige con la final y no con una sola vuelta porque hay pilotos que completan la
primera vuelta y se quedan detenidos en la segunda o la tercera.
"""
from collections import deque
from pathlib import Path
import csv
import json
import time

import numpy as np
import torch

from mi_configuracion import CONFIG
import mi_recompensa
import mi_algoritmo as ppo

from src.carrera.limites import validar_hiperparametros, resumen_reglas
from src.carrera.pistas import ORDEN_PISTAS
from src.carrera.entorno import EntornoVectorial
from src.carrera.red import Actor, Critico, aplanar_actor
from src.carrera.politica import PoliticaNumpy
from src.carrera.evaluacion import evaluar_politica, imprimir_tabla, VUELTAS_FINAL, TIEMPO_FINAL
from src.carrera.exportar import guardar_piloto_npy

CARPETA_SALIDA = Path("salidas")
VUELTAS_ENTRENAMIENTO = 1
TIEMPO_ENTRENAMIENTO = 24.0
NUM_SENSORES = 12


def _promedio(resumenes, clave):
    return float(np.mean([r[clave] for r in resumenes])) if resumenes else float("nan")


def entrenar(config=None, recompensa=None, verbose=True, carpeta_salida=CARPETA_SALIDA):
    """Ejecuta el ciclo completo de PPO y devuelve (mejor_vector, mejor_puntaje, historial)."""
    cfg = dict(CONFIG if config is None else config)
    recompensa = mi_recompensa.recompensa if recompensa is None else recompensa
    validar_hiperparametros(cfg)
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["semilla"])
    rng = np.random.default_rng(cfg["semilla"])
    entorno = EntornoVectorial(
        ORDEN_PISTAS, VUELTAS_ENTRENAMIENTO, TIEMPO_ENTRENAMIENTO,
        cfg["inicio_aleatorio"], cfg["semilla"],
    )
    n_pistas = len(entorno)
    T = cfg["pasos_por_rollout"]
    pasos_por_iteracion = T * n_pistas
    n_iteraciones = cfg["pasos_totales"] // pasos_por_iteracion

    actor, critico = Actor(), Critico()
    parametros = list(actor.parameters()) + list(critico.parameters())
    optimizador = torch.optim.Adam(parametros, lr=cfg["tasa_aprendizaje"], eps=1e-5)

    obs = entorno.reset()
    episodios_recientes = deque(maxlen=50)
    historial = []
    mejor_puntaje = -np.inf
    mejor_vector = None
    inicio = time.time()

    for iteracion in range(n_iteraciones):
        # ---------------------------------------------------------------- 1. recolección
        buf_obs = np.zeros((T, n_pistas, NUM_SENSORES))
        buf_acciones = np.zeros((T, n_pistas), dtype=np.int64)
        buf_logp = np.zeros((T, n_pistas))
        buf_valores = np.zeros((T, n_pistas))
        buf_valores_sig = np.zeros((T, n_pistas))
        buf_recompensas = np.zeros((T, n_pistas))
        buf_fin = np.zeros((T, n_pistas))
        with torch.no_grad():
            for t in range(T):
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                distribucion = actor.distribucion(obs_t)
                acciones = distribucion.sample()
                buf_obs[t] = obs
                buf_acciones[t] = acciones.numpy()
                buf_logp[t] = distribucion.log_prob(acciones).numpy()
                buf_valores[t] = critico(obs_t).numpy()

                obs, recompensas, fin, terminal, bootstrap, resumenes = entorno.step(acciones.numpy(), recompensa)
                valores_bootstrap = critico(torch.as_tensor(bootstrap, dtype=torch.float32)).numpy()
                buf_valores_sig[t] = valores_bootstrap * (1.0 - terminal)   # desde la meta no queda futuro
                buf_recompensas[t] = recompensas
                buf_fin[t] = fin
                episodios_recientes.extend(resumenes)

        # ---------------------------------------------------------------- 2. ventajas y objetivos
        ventajas = ppo.calcular_ventajas(
            buf_recompensas, buf_valores, buf_valores_sig, buf_fin, cfg["gamma"], cfg["lam"],
        )
        retornos_objetivo = ventajas + buf_valores

        B = T * n_pistas
        obs_b = torch.as_tensor(buf_obs.reshape(B, NUM_SENSORES), dtype=torch.float32)
        acciones_b = torch.as_tensor(buf_acciones.reshape(B))
        logp_viejo_b = torch.as_tensor(buf_logp.reshape(B), dtype=torch.float32)
        ventajas_b = torch.as_tensor(ppo.normalizar_ventajas(ventajas.reshape(B)), dtype=torch.float32)
        retornos_b = torch.as_tensor(retornos_objetivo.reshape(B), dtype=torch.float32)

        # ---------------------------------------------------------------- 3. K épocas de optimización
        indices = np.arange(B)
        M = cfg["tam_minibatch"]
        suma_actor = suma_critico = suma_recorte = suma_kl = 0.0
        n_minibatches = 0
        for _ in range(cfg["epocas"]):
            rng.shuffle(indices)
            for comienzo in range(0, B, M):
                idx = torch.as_tensor(indices[comienzo:comienzo + M])
                distribucion = actor.distribucion(obs_b[idx])
                logp_nuevo = distribucion.log_prob(acciones_b[idx])
                razon = torch.exp(logp_nuevo - logp_viejo_b[idx])

                l_actor = ppo.perdida_actor(razon, ventajas_b[idx], cfg["epsilon_clip"])
                l_critico = ppo.perdida_critico(critico(obs_b[idx]), retornos_b[idx])
                entropia = distribucion.entropy().mean()
                perdida = l_actor + cfg["coef_valor"] * l_critico - cfg["coef_entropia"] * entropia

                optimizador.zero_grad()
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(parametros, 0.5)
                optimizador.step()

                with torch.no_grad():
                    suma_actor += float(l_actor)
                    suma_critico += float(l_critico)
                    suma_recorte += float(((razon - 1.0).abs() > cfg["epsilon_clip"]).float().mean())
                    suma_kl += float((logp_viejo_b[idx] - logp_nuevo).mean())
                n_minibatches += 1

        # ---------------------------------------------------------------- 4. prueba oficial (la final) y registro
        pasos = (iteracion + 1) * pasos_por_iteracion
        puntaje = float("nan")
        if (iteracion + 1) % cfg["evaluar_cada"] == 0 or iteracion == n_iteraciones - 1:
            vector = aplanar_actor(actor)
            _, puntaje = evaluar_politica(PoliticaNumpy(vector), VUELTAS_FINAL, TIEMPO_FINAL)
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_vector = vector.copy()
                np.save(carpeta_salida / "mejor_actor.npy", mejor_vector)

        fila = {
            "iteracion": iteracion,
            "pasos": pasos,
            "retorno_medio": _promedio(episodios_recientes, "retorno"),
            "vueltas_media": _promedio(episodios_recientes, "vueltas"),
            "checkpoints_media": _promedio(episodios_recientes, "checkpoints"),
            "tiempo_fuera_media": _promedio(episodios_recientes, "tiempo_fuera"),
            "colisiones_media": _promedio(episodios_recientes, "colisiones"),
            "penalizacion_media": _promedio(episodios_recientes, "tiempo_penalizacion"),
            "perdida_actor": suma_actor / n_minibatches,
            "perdida_critico": suma_critico / n_minibatches,
            "fraccion_recortada": suma_recorte / n_minibatches,
            "kl_aprox": suma_kl / n_minibatches,
            "puntaje_oficial": puntaje,
            "mejor_puntaje": mejor_puntaje if np.isfinite(mejor_puntaje) else float("nan"),
        }
        historial.append(fila)

        if verbose:
            oficial = f"{puntaje:8.1f}" if np.isfinite(puntaje) else "    -   "
            print(
                f"It {iteracion:03d} | pasos {pasos:7d} | retorno {fila['retorno_medio']:8.2f} | "
                f"vueltas {fila['vueltas_media']:4.2f} | CP {fila['checkpoints_media']:5.1f} | "
                f"fuera {fila['tiempo_fuera_media']:5.2f}s | choques {fila['colisiones_media']:4.2f} | "
                f"castigo {fila['penalizacion_media']:5.2f}s | recorte {fila['fraccion_recortada']:4.2f} | oficial {oficial} | mejor {fila['mejor_puntaje']:8.1f}"
            )

    np.save(carpeta_salida / "ultimo_actor.npy", aplanar_actor(actor))
    with (carpeta_salida / "historial.csv").open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=historial[0].keys())
        writer.writeheader()
        writer.writerows(historial)
    (carpeta_salida / "configuracion_usada.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nEntrenamiento terminado en {time.time() - inicio:.0f} s.")
        print(f"Prueba oficial del mejor actor ({VUELTAS_FINAL} vueltas o {TIEMPO_FINAL:.0f} s por pista):")
        resultados, _ = evaluar_politica(PoliticaNumpy(mejor_vector), VUELTAS_FINAL, TIEMPO_FINAL)
        imprimir_tabla(resultados)
        print("Actor temporal: salidas/mejor_actor.npy")
    return mejor_vector, mejor_puntaje, historial


def exportar_entrega(nombre_piloto, pesos):
    """Crea el .npy que debe recibir el profesor."""
    return guardar_piloto_npy(nombre_piloto, pesos, carpeta="entregas")


if __name__ == "__main__":
    resumen_reglas()
    mejor, _, _ = entrenar()
    print("\nCuando estes satisfecho con tu piloto:")
    print("  python exportar_mi_piloto.py      (edita antes NOMBRE_PILOTO)")
    print("  python verificar_entrega.py entregas/<Nombre>.npy")
