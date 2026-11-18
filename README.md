# 🏁 UIS Grand Prix Foundationals Models

Plataforma educativa competitiva de conducción autónoma para comparar **In-Context Learning (ICL con KNN ponderado / aleatorio)** contra **Fine-Tuning con PEFT/LoRA** sobre un único modelo base **`Qwen/Qwen2.5-1.5B-Instruct`**.

---

## 1. Arquitectura Cliente–Servidor

```text
grand_prix/
├── backend_server/               # GPU Server (RTX 3060 local o Colab A100/L4)
│   ├── .env / .env.example       # Modelo base, límites anti-trampa y peso de latencia
│   ├── requirements.txt
│   ├── motor_qwen.py             # Qwen2.5-1.5B-Instruct (bfloat16 + SDPA) + KNN + LoRA + gpu_lock
│   ├── server.py                 # API FastAPI + Leaderboard HTML en vivo en /ranking
│   └── adapters_cache/           # Pesos LoRA aislados por student_id (ignorado en Git)
│
└── cliente_estudiante/           # PC del estudiante (CPU + Pygame)
    ├── .env / .env.example       # SERVER_URL, STUDENT_ID, STUDENT_NAME, timeouts
    ├── requirements.txt
    ├── config_estudiante.py      # Panel único de hiperparámetros (ICL vs LORA)
    ├── bridge.py                 # Cliente HTTP con reintentos y medición real de RTT (ms)
    ├── juego_pistas.py           # Simulador Pygame (física RL_cars, P1–P3 y B1–B3)
    └── data/mis_jugadas.json     # Demostraciones grabadas por el estudiante (120 semilla)
```

---

## 2. Mecánica de Competencia y Variables del Modelo

### Variables de Entrada (Foundational Model)
1. **5 rayos de distancia normalizados (`[0..1]`):** `[izq, diag_izq, frente, diag_der, der]` (activos tanto dentro como fuera de la pista).
2. **`cp_dist` (`[0..1]`):** distancia normalizada al próximo checkpoint (dibujada con rayo azul celeste en pista).
3. **`velocidad` (`[0..1]`):** rapidez actual normalizada respecto a `7.6 m/s`.
4. **`Historial`:** secuencia de las últimas `ventana_historia` acciones (por defecto **5** movimientos, mostrados con íconos en el panel).

### Variable de Salida (1 Token)
- **`0`:** Recto + Acelerar (`↑` / `W`)
- **`1`:** Izquierda + Acelerar (`←` / `A`)
- **`2`:** Derecha + Acelerar (`→` / `D`)
- **`3`:** Frenar (`↓` / `S`, o al soltar todas las teclas en modo manual)

### Formato de Prompt Ultra-Compacto
```text
S:<s1>,<s2>,<s3>,<s4>,<s5>|C:<cp_dist>|V:<vel>|H:<historial> -> <accion>
```

### Circuitos y Penalización por Latencia
- **Pistas de Entrenamiento (`P1`, `P2`, `P3`):** Óvalo, Curvas Amplias y Chicane Técnica para grabar jugadas y probar el piloto.
- **Pistas Ocultas de Evaluación (`B1`, `B2`, `B3`):** La Serpiente, La Cantera (obstáculos fijos) y Hora Pico (obstáculos móviles).
- **Sincronización y Latencia:** En modo autónomo, cada `RECORD_EVERY_N_FRAMES` el simulador espera sincrónicamente la respuesta HTTP del servidor y acumula la latencia real (`rtt_ms`) al reloj penalizado (`LATENCY_PENALTY_WEIGHT=1.5`).
- **Límite del recuadro:** Si el carro alcanza el borde de la pantalla en modo manual o autónomo, rebota automáticamente apuntando al próximo checkpoint.

---

## 3. Hiperparámetros del Estudiante (`config_estudiante.py`)

| Sección | Parámetro | Descripción |
|---|---|---|
| General | `ESTRATEGIA` | `"ICL"` (prompting con ejemplos) o `"LORA"` (fine-tuning PEFT) |
| `CONFIG_DATOS` | `ventana_historia` | Cantidad de acciones previas en `H` (`1..8`, por defecto `5`) |
| `CONFIG_DATOS` | `precision_sensores` | Bins de discretización entera (`20`, `50`, `100`, etc.) |
| `CONFIG_ICL` | `k_shots` | Ejemplos en el prompt (`1..30`) |
| `CONFIG_ICL` | `seleccion` | `"knn"` (vecinos más cercanos) o `"random"` |
| `CONFIG_ICL` | `pesos_sensores` | Pesos `[izq, diag_izq, frente, diag_der, der]` para KNN |
| `CONFIG_ICL` | `temperature`, `top_p` | Control de muestreo (`0.0` = determinista) |
| `CONFIG_LORA` | `num_ejemplos`, `epochs`, `learning_rate` | Presupuesto de entrenamiento (máx `15` épocas) |
| `CONFIG_LORA` | `lora_r`, `lora_alpha`, `target_modules` | Rango LoRA (`2..32`) y capas (`q_proj`, `v_proj`, MLP, etc.) |

---

## 4. Ejecución Local (Windows / RTX 3060)

### Paso A: Levantar el Backend (`grand_prix/backend_server/`)

```bat
cd /d "grand_prix\backend_server"
pip install -r requirements.txt
copy .env.example .env
uvicorn server:app --host 0.0.0.0 --port 8000
```

*(Si usas un entorno Conda específico en Windows como `grandprix`, ejecuta directamente `"C:\Users\<usuario>\miniconda3\envs\grandprix\python.exe" -m uvicorn server:app --host 0.0.0.0 --port 8000`).*

- Salud del servidor y GPU: `http://127.0.0.1:8000/health`
- Tabla de posiciones en vivo (auto-refresco cada 5 s): `http://127.0.0.1:8000/ranking`

> **Modo simulado en CPU sin descargar Qwen (opcional):** establece `MOCK_LLM=1` (`set MOCK_LLM=1` en CMD o `$env:MOCK_LLM="1"` en PowerShell).

### Paso B: Ejecutar el Cliente del Estudiante (`grand_prix/cliente_estudiante/`)

```bat
cd /d "grand_prix\cliente_estudiante"
pip install -r requirements.txt
copy .env.example .env
python juego_pistas.py
```

#### Flujo en el Simulador
1. Edita `grand_prix/cliente_estudiante/.env` con tu `STUDENT_ID` y `STUDENT_NAME`.
2. **`[1]` Grabar Jugadas (`P1`, `P2`, `P3`):**
   - Selecciona la pista con los botones **`P1` | `P2` | `P3`**, ajusta la frecuencia (`Cada N frames`), pulsa **`INICIAR GRABACION`** y conduce con **Flechas** o **`W/A/S/D`** (`R` reinicia al inicio).
   - Pulsa **`Ver Frames`** para inspeccionar visualmente cada muestra (`A/D` o `←/→`), eliminar frames individuales, borrar por pista (`borrar-p1`, `borrar-p2`, `borrar-p3`) o reiniciar todo (`borrar-todo`).
   - Pulsa **`Guardar y Salir`** (o `ESC`) para guardar en `data/mis_jugadas.json`.
3. Edita `grand_prix/cliente_estudiante/config_estudiante.py` para configurar `ICL` o `LORA`.
4. Si usas `LORA`, ejecuta **`[2]` Entrenar Piloto**.
5. Prueba en **`[3]` Modo Autónomo - Prueba (`P1`, `P2`, `P3`)**.
6. Compite en **`[4]` Evaluación Oficial (`B1`, `B2`, `B3`)** para enviar tu puntaje a `/ranking`.

#### Comandos CLI sin ventana gráfica
```bash
python juego_pistas.py --generar-semilla      # Regenera 120 jugadas semilla en P1/P2/P3
python juego_pistas.py --modo 2 --headless    # Entrenar LoRA desde consola
python juego_pistas.py --modo 4 --headless    # Evaluación oficial B1/B2/B3
```

---

## 5. Despliegue en Clase: Google Colab Pro (A100 / L4) + Cloudflare Tunnel

Para atender hasta **12 estudiantes en paralelo** sin modificar el backend:

### Celda 1: Instalar dependencias y `cloudflared`
```python
!nvidia-smi
!pip -q install -r /content/grand_prix/backend_server/requirements.txt

!wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /usr/local/bin/cloudflared
```

### Celda 2: Iniciar FastAPI (`--workers 1`) y abrir túnel `trycloudflare.com`
```python
import os, re, shutil, subprocess, time, urllib.request

os.chdir("/content/grand_prix/backend_server")
if not os.path.exists(".env") and os.path.exists(".env.example"):
    shutil.copy(".env.example", ".env")

server_log = open("/content/uvicorn.log", "w")
server_proc = subprocess.Popen(
    ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"],
    stdout=server_log,
    stderr=subprocess.STDOUT,
)

print("Cargando Qwen2.5-1.5B-Instruct en GPU...")
for _ in range(90):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
            if r.status == 200:
                print("Backend listo:", r.read().decode())
                break
    except Exception:
        time.sleep(2)

cf_log = open("/content/cloudflared.log", "w")
cf_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
    stdout=cf_log,
    stderr=subprocess.STDOUT,
)

public_url = None
for _ in range(30):
    time.sleep(1)
    txt = open("/content/cloudflared.log", errors="ignore").read()
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
    if m:
        public_url = m.group(0)
        break

print("\n" + "=" * 68)
print(f"  SERVER_URL={public_url}")
print(f"  RANKING EN VIVO: {public_url}/ranking")
print("=" * 68)
```

Cada estudiante actualiza `SERVER_URL` en su `grand_prix/cliente_estudiante/.env` y ejecuta `python juego_pistas.py`.

---

## 6. Endpoints del Backend (`grand_prix/backend_server/server.py`)

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado de GPU, VRAM libre, modelo base cargado y límites del `.env` |
| `POST` | `/entrenar_lora` | Valida límites (`MAX_LORA_R`, `MAX_LORA_EPOCHS`) y entrena adaptador LoRA en `adapters_cache/{student_id}` |
| `POST` | `/decidir_accion` | Ejecuta ICL (`knn`/`random`) o LoRA (`set_adapter`) y devuelve `{"accion": int, "latencia_gpu_ms": float}` |
| `POST` | `/registrar_puntaje` | Calcula puntaje oficial de Pistas `B1, B2, B3` penalizando latencia y actualiza `ranking.json` |
| `GET` | `/ranking` | Tabla HTML en vivo con auto-refresco cada 5 segundos |
| `GET` | `/ranking/json` | Tabla de posiciones en formato JSON |
