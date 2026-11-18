# Car Racing PPO Challenge — Estudiantes

El mismo carro, los mismos 12 sensores y las mismas cinco pistas del reto genético, más una sexta
con obstáculos en movimiento. Ahora el piloto es una red neuronal (12 → 32 → 32 → 7) entrenada con **PPO**, y tu trabajo
ya no es diseñar una *fitness* para toda la carrera, sino una **recompensa para cada paso de 0.06 s**.

Tu entrega final es un único `.npy` con 1703 pesos. `Hamilton.npy` aparecerá como `hamilton` en el torneo.

## Edita
- `mi_recompensa.py` — la recompensa por paso (lo más importante)
- `mi_configuracion.py` — los hiperparámetros de PPO
- `mi_algoritmo.py` — opcional: GAE, normalización de ventajas, pérdida recortada, pérdida del crítico

## Ejecuta
```bash
pip install -r requirements.txt
python tests/smoke_test.py                       # todo funciona
python entrenar.py                               # entrena y guarda salidas/mejor_actor.npy
python exportar_mi_piloto.py                     # edita antes NOMBRE_PILOTO -> entregas/<Nombre>.npy
python verificar_entrega.py entregas/Hamilton.npy          # prueba corta: 1 vuelta o 24 s
python verificar_entrega.py entregas/Hamilton.npy --final  # la final: 5 vueltas o 150 s (la prueba oficial y el torneo)
```
Después abre los HTML de `web/`: `como_ver_el_problema.html` explica el reto y
`visualizador_piloto.html` corre tu `.npy` mostrando en vivo la distribución π(a | s).

## Qué cambia respecto al reto genético
| | Reto genético | Este reto |
|---|---|---|
| el piloto | 36 genes: un controlador lineal | 1703 pesos: actor 12 → 32 → 32 → 7 (softmax) |
| la acción | tres números continuos | una de 7 maniobras discretas |
| lo que diseñas | una fitness al final de la carrera | una recompensa después de cada paso |
| cómo se aprende | selección, cruce, mutación | actor–crítico, ventaja, objetivo recortado |
| el presupuesto | población × generaciones ≤ 3000 | pasos de interacción ≤ 600 000 |

## Las 7 maniobras
`acelerar + izquierda`, `acelerar + recto`, `acelerar + derecha`, `rodar + izquierda`, `rodar + recto`,
`rodar + derecha`, `frenar`. El volante siempre va a tope (−1, 0 o +1); "rodar" es soltar el acelerador.

## Las 6 pistas
`1 - Óvalo`, `2 - Curvas largas`, `3 - Chicane`, `4 - Serpiente`, `5 - La cantera` (obstáculos fijos) y
`6 - Hora pico`: un trazado amable que cinco obstáculos cruzan de borde a borde, cada uno con su propio ritmo
(periodos de 3.6 a 5.5 s). Su posición depende solo del tiempo del episodio, así que la prueba oficial sigue
siendo determinista. Los sensores son los mismos 12: el piloto ve la distancia y la dirección del obstáculo
más cercano en cada instante, pero no su velocidad; para anticiparse tiene que aprender a leer la escena.

## Castigos de tiempo
Chocar y salirse **suman segundos al reloj de carrera**, el mismo reloj que agota el límite de la prueba (24 s o 150 s):

| falta | castigo |
|---|---|
| cada choque con un obstáculo | +2 s |
| cada segundo fuera de la pista | +1 s (el reloj corre al doble) |

Un piloto que choca no solo pierde puntos: le queda menos tiempo para completar sus vueltas. Un choque cuenta una
sola vez aunque el carro siga pegado al obstáculo (empujando contra él o arrastrado por uno que se mueve).
En tu recompensa tienes `paso.penalizacion` (los segundos que sumó ese paso) y `paso.tiempo_carrera` (el reloj oficial).

## Límites
| hiperparámetro | rango |
|---|---|
| `pasos_totales` | 20 000 – 600 000 |
| `pasos_por_rollout` | 128 – 1024 (por pista; las 6 pistas corren en paralelo) |
| `epocas` | 1 – 10 |
| `tam_minibatch` | 32 – 2048, y no mayor que `pasos_por_rollout × 6` |
| `tasa_aprendizaje` | 5e-5 – 3e-3 |
| `gamma` | 0.90 – 0.999 |
| `lam` | 0.80 – 1.00 |
| `epsilon_clip` | 0.05 – 0.40 |
| `coef_valor` (c₁) | 0.10 – 1.00 |
| `coef_entropia` | 0.00 – 0.05 (opcional; 0 = la fórmula vista en clase) |
| `evaluar_cada` | 1 – 50 |

Los pesos entregados deben ser 1703 números finitos en [−50, 50].

## La prueba oficial y el torneo
- **Entrenamiento:** episodios de 1 vuelta o 24 s de reloj de carrera; si `inicio_aleatorio` es `True`, el carro sale de un checkpoint al azar.
- **Prueba oficial** (la que elige `mejor_actor.npy`) **y torneo:** la final, 5 vueltas o 150 s de reloj de carrera por
  pista, salida desde la meta, conducción determinista (siempre la maniobra más probable). Se elige con la final y no con
  una sola vuelta porque hay pilotos que completan la primera vuelta y luego se quedan detenidos: una vuelta no lo ve.
- **Puntaje por pista:** `1000·vueltas + 50·checkpoints − reloj de carrera`, donde
  `reloj de carrera = tiempo + 1·tiempo_fuera + 2·choques`. Se suman las seis pistas.

## Qué deberías ver
- Con la recompensa de partida (acercarse al siguiente checkpoint) el carro termina las cinco vueltas en casi todas las
  pistas, pero a un costo enorme: cruzar un checkpoint le "cuesta" porque el siguiente está lejos, así que frena, se sale
  o choca. Pasa unos 45 s fuera de la pista en Chicane y en Serpiente, choca una decena de veces en La cantera y en
  Serpiente no alcanza la quinta vuelta. La prueba oficial se queda alrededor de 45 000 puntos (varía con la semilla).
  Ese es el punto de partida que debes mejorar.
- Cinco vueltas limpias y rápidas en las seis pistas valen unos 47 000 puntos. Lo que queda por ganar está en completar
  Serpiente (unos 1 250 puntos) y en no pagar castigos: cada segundo fuera de la pista y cada choque restan puntos y,
  además, gastan reloj. Con una recompensa bien pensada, la prueba oficial empieza a mostrar vueltas completas en las
  pistas fáciles alrededor de los 50 000 – 100 000 pasos.
- Un entrenamiento de 300 000 pasos tarda pocos minutos en un portátil (solo CPU).
- Columnas útiles de `historial.csv`: `puntaje_oficial` (la final, cada `evaluar_cada` actualizaciones);
  `checkpoints_media`, `tiempo_fuera_media` y `penalizacion_media` (segundos de castigo por episodio) cuentan si el
  carro conduce;
  `fraccion_recortada` y `kl_aprox` cuentan cuánto cambia la política en cada actualización (si son 0, no está aprendiendo;
  si son muy grandes, está dando saltos demasiado largos).

## Archivos que genera `entrenar.py`
`salidas/mejor_actor.npy` (el mejor según la prueba oficial), `salidas/ultimo_actor.npy`,
`salidas/historial.csv` (retorno, vueltas, checkpoints, tiempo fuera, choques, castigo, pérdidas, fracción recortada, KL) y
`salidas/configuracion_usada.json`.
