# Prompt de Construcción MVP: UIS Grand Prix Foundationals models

Actúa como un Ingeniero Senior de Machine Learning y Arquitecto de Software en Python. Vamos a construir desde cero el MVP completo de una plataforma educativa competitiva para una clase de Modelos Fundacionales llamada **UIS Grand Prix Foundationals models**.

El sistema tiene una arquitectura **Cliente-Servidor** desacoplada:
1. **Desarrollo y pruebas locales:** Correrá primero en mi estación de trabajo local con una GPU **NVIDIA RTX 3060 Ti**.
2. **Despliegue en clase:** Se migrará sin cambiar una sola línea de lógica a **Google Colab Pro (NVIDIA A100 40 GB)** expuesto mediante un túnel de Cloudflare (`trycloudflare.com`) para atender hasta **12 estudiantes en paralelo**.

> **NOTA SOBRE EL SIMULADOR EXISTENTE:** Revisa el código de mi juego de carreras que RL_cars. Extrae y reutiliza su motor físico, renderizado y lógica de colisiones para integrarlo en el módulo del cliente (`cliente_estudiante/`), adaptándolo a los requisitos de sensores, pistas y puente HTTP descritos abajo.

---

## 1. Estructura de Carpetas del Proyecto

Crea exactamente esta estructura de directorios:

```text
fewshot_grand_prix/
├── backend_server/
│   ├── .env
│   ├── .env.example
│   ├── requirements.txt
│   ├── server.py               # API FastAPI + dashboard HTML simple en /ranking
│   ├── motor_qwen.py           # Carga de Qwen2.5-1.5B, motor ICL (KNN) y entrenamiento PEFT/LoRA
│   └── adapters_cache/         # Carpeta donde se guardan los pesos LoRA por student_id
│
└── cliente_estudiante/
    ├── .env
    ├── .env.example
    ├── requirements.txt
    ├── config_estudiante.py    # Panel único de hiperparámetros (ICL vs LoRA) que edita el alumno
    ├── bridge.py               # Cliente HTTP con manejo de timeouts, reintentos y medición de latencia
    ├── juego_pistas.py         # Simulador integrado con 3 Pistas A (Train) y 3 Pistas B (Test)
    └── data/                   # Donde se guarda mis_jugadas.json localmente
```

---

## 2. Configuración de Variables de Entorno (`.env`)

Genera ambos archivos `.env` y sus respectivos `.env.example` usando `python-dotenv`:

### `backend_server/.env`
```env
HOST=0.0.0.0
PORT=8000
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
TORCH_DTYPE=bfloat16
DEVICE=cuda
ADAPTERS_DIR=./adapters_cache
MAX_K_SHOTS=30
MAX_LORA_EPOCHS=15
MAX_LORA_R=32
LATENCY_PENALTY_WEIGHT=1.5
```

### `cliente_estudiante/.env`
```env
# En local (3060 Ti) usa localhost; en clase se cambia por la URL de trycloudflare.com
SERVER_URL=http://127.0.0.1:8000
STUDENT_ID=estudiante_01
STUDENT_NAME=Oscar_Prueba
REQUEST_TIMEOUT_SEC=5.0
RECORD_EVERY_N_FRAMES=6
```

---

## 3. Especificaciones del Backend (`backend_server/`)

1. **Modelo Base Único:** Carga `Qwen/Qwen2.5-1.5B-Instruct` en `bfloat16` con `attn_implementation="sdpa"`.
2. **Aislamiento y Concurrencia (`threading.Lock`):**
   - Implementa un `threading.Lock()` global en `motor_qwen.py` para proteger las operaciones críticas en GPU (entrenamiento LoRA y cambio de adaptadores activos) de modo que 12 estudiantes concurrentes nunca mezclen pesos ni generen errores de CUDA.
   - Gestiona los adaptadores PEFT por `student_id`: al entrenar, guarda el adaptador en `ADAPTERS_DIR/{student_id}` y cárgalo dinámicamente usando `PeftModel` (`load_adapter` / `set_adapter` / `disable_adapter` para modo ICL).
3. **Formato de Prompt Ultra-Compacto (1 Token de Salida):**
   - Cada muestra se formatea como: `S:<s1>,<s2>,<s3>,<s4>,<s5>|H:<historial> -> <accion>` donde las acciones son un único dígito (`0`: recto, `1`: izquierda, `2`: derecha, `3`: frenar).
   - Configura `model.generate()` con `max_new_tokens=2`, `do_sample=(temperature > 0)` y un parser robusto con *fallback* a acción `0` (o freno suave `3`) si el modelo llega a emitir un carácter no numérico.
4. **Endpoints FastAPI en `server.py`:**
   - `GET /health`: Verifica GPU activa, VRAM libre y modelo cargado.
   - `POST /entrenar_lora`: Recibe `student_id`, `jugadas` (lista de ejemplos) e hiperparámetros (`lora_r`, `lora_alpha`, `target_modules`, `epochs`, `learning_rate`, `num_ejemplos`). Valida límites anti-trampa del `.env`, entrena el adaptador LoRA y devuelve tiempo de entrenamiento y loss final.
   - `POST /decidir_accion`: Recibe `student_id`, `estrategia` (`"ICL"` o `"LORA"`), `estado_actual`, `historial`, `jugadas_referencia` (si es ICL) e hiperparámetros (`k_shots`, `seleccion`, `pesos_sensores`, `temperature`, `top_p`, `precision_sensores`). Si es ICL, ejecuta la búsqueda KNN ponderada o muestreo aleatorio en el servidor, arma el prompt y predice la acción. Devuelve `{"accion": int, "latencia_gpu_ms": float}`.
   - `POST /registrar_puntaje`: Recibe resultados de las 3 Pistas B (`B1`, `B2`, `B3`), calcula el puntaje oficial penalizando el tiempo total de latencia acumulada y actualiza `ranking.json`.
   - `GET /ranking`: Devuelve una vista HTML limpia con auto-refresco cada 5 segundos (para proyectar en clase) mostrando la tabla de posiciones en vivo (Posición, Estudiante, Estrategia, % Promedio Pistas B, Tiempo Total, Latencia Promedio ms, Score Final).

---

## 4. Especificaciones del Cliente (`cliente_estudiante/`)

1. **`config_estudiante.py` (Panel del Alumno):**
   Debe contener diccionarios claros y comentados para que el alumno experimente:
   - `ESTRATEGIA`: `"ICL"` o `"LORA"`.
   - `CONFIG_DATOS`: `ventana_historia` (1 a 4 frames), `precision_sensores` (ej. 20, 50, 100 bins de discretización).
   - `CONFIG_ICL`: `k_shots` (1 a 30), `seleccion` (`"knn"` o `"random"`), `pesos_sensores` (lista de 5 floats para ponderar distancias en KNN), `temperature`, `top_p`.
   - `CONFIG_LORA`: `num_ejemplos`, `lora_r` (2, 4, 8, 16, 32), `lora_alpha`, `target_modules` (ej. `["q_proj", "v_proj"]` vs incluir MLP `["gate_proj", "up_proj", "down_proj"]`), `epochs`, `learning_rate`.

2. **`juego_pistas.py` (Entorno de Carreras):**
   - **6 Circuitos:** 3 pistas de entrenamiento (`A1`, `A2`, `A3`) con dificultad progresiva (rectas, curvas amplias, chicanes) y 3 pistas ocultas/de evaluación (`B1`, `B2`, `B3`) con trazados distintos para evaluar generalización real.
   - **Sensores y Acciones:** 5 rayos de distancia normalizados (`izq`, `diag_izq`, `frente`, `diag_der`, `der`) y 4 acciones discretas (`0`: acelerar recto, `1`: girar izquierda + acelerar, `2`: girar derecha + acelerar, `3`: frenar).
   - **Modos de Ejecución (Menú en pantalla o CLI):**
     - `[1] Grabar Jugadas (Pistas A1, A2, A3)`: Conducción con teclado; captura 1 muestra cada `RECORD_EVERY_N_FRAMES` frames y guarda en `data/mis_jugadas.json`.
     - `[2] Entrenar Piloto (Solo si ESTRATEGIA == "LORA")`: Llama a `bridge.entrenar_lora()`.
     - `[3] Modo Autónomo - Prueba (Pistas A)`: Corre el auto consultando a `bridge.decidir_accion()`.
     - `[4] Evaluación Oficial (Pistas B1, B2, B3)`: Corre secuencialmente las 3 pistas B, acumula progreso, tiempo de vuelta y latencia real de red+GPU, y envía el resultado final a `/registrar_puntaje`.
   - **Sincronización Física:** En modo autónomo, cada vez que toque consultar al servidor (cada `RECORD_EVERY_N_FRAMES`), el paso físico del auto espera la respuesta HTTP para evitar choques injustos por fluctuaciones de Wi-Fi, pero suma el tiempo real de espera (`rtt_ms`) al cronómetro penalizado de la carrera.

---

## 5. Plan de Entrega

1. Inspecciona primero mi archivo del juego de autos actual en el proyecto (si aún no te lo he mostrado, pídemelo antes de escribir `juego_pistas.py` o crea una implementación limpia en Pygame modularizada para reemplazar fácilmente la física).
2. Genera todos los archivos de `backend_server/` y `cliente_estudiante/` completos, tipados y listos para ejecutar.
3. Incluye al final un `README.md` con los comandos exactos para:
   - Levantar el servidor y cliente localmente en mi **RTX 3060 Ti**.
   - Ejecutar el mismo `backend_server/` en una celda de **Google Colab Pro (A100)** con `cloudflared` para el día de la clase.