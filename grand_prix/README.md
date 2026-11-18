# 🏁 UIS Grand Prix Foundationals Models (MVP)

Plataforma educativa competitiva de conducción autónoma para comparar **In-Context Learning (ICL con KNN ponderado / aleatorio)** contra **Fine-Tuning con PEFT/LoRA** sobre un único modelo base **`Qwen/Qwen2.5-1.5B-Instruct`**.

---

## 1. Arquitectura Cliente–Servidor

```text
grand_prix/
├── backend_server/               # GPU Server (RTX 3060 Ti local o Colab A100/L4)
│   ├── .env / .env.example       # Configuración del modelo, límites anti-trampa y penalización
│   ├── requirements.txt
│   ├── motor_qwen.py             # Qwen2.5-1.5B-Instruct (bfloat16 + SDPA) + KNN + LoRA + Lock GPU
│   ├── server.py                 # API FastAPI + Leaderboard HTML en /ranking
│   └── adapters_cache/           # Pesos LoRA aislados por student_id
│
└── cliente_estudiante/           # PC del estudiante (solo CPU + Pygame)
    ├── .env / .env.example       # SERVER_URL, STUDENT_ID, STUDENT_NAME, timeouts
    ├── requirements.txt
    ├── config_estudiante.py      # Panel único de hiperparámetros (ICL vs LORA)
    ├── bridge.py                 # Cliente HTTP con reintentos y medición real de RTT
    ├── juego_pistas.py           # Simulador Pygame (física RL_cars, 3 Pistas A + 3 Pistas B)
    └── data/mis_jugadas.json     # Demostraciones grabadas por el estudiante
```

### Mecánica de Competencia
- **Sensores (5 rayos normalizados `[0..1]`):** `[izq, diag_izq, frente, diag_der, der]`.
- **Acciones discretas (`1` token):** `0` = recto + acelerar, `1` = izquierda + acelerar, `2` = derecha + acelerar, `3` = frenar.
- **Formato de prompt ultra-compacto:** `S:<s1>,<s2>,<s3>,<s4>,<s5>|H:<historial> -> <accion>`.
- **Pistas A (`A1`, `A2`, `A3`):** Circuitos de entrenamiento para grabar jugadas y probar en local.
- **Pistas B (`B1`, `B2`, `B3`):** Circuitos ocultos de evaluación oficial (incluyen serpiente, obstáculos fijos y tráfico móvil).
- **Penalización por latencia:** Cada consulta al servidor detiene sincrónicamente el paso físico hasta recibir respuesta y acumula el tiempo real de red + GPU (`rtt_ms`) al reloj penalizado del ranking (`LATENCY_PENALTY_WEIGHT=1.5`).

---

## 2. Ejecución Local (Estación de Trabajo NVIDIA RTX 3060 Ti)

### Paso A: Levantar el Backend (`backend_server/`)

Abre una terminal en `grand_prix/backend_server/`:

```bash
cd grand_prix/backend_server
pip install -r requirements.txt
cp .env.example .env   # (en Windows PowerShell: Copy-Item .env.example .env)
uvicorn server:app --host 0.0.0.0 --port 8000
```

Verifica en tu navegador o terminal que la GPU y el modelo quedaron activos:
- Salud del servidor: `http://127.0.0.1:8000/health`
- Ranking en vivo (auto-refresco cada 5 s): `http://127.0.0.1:8000/ranking`

> **Nota para pruebas rápidas en CPU sin descargar Qwen (opcional):**
> Puedes iniciar el servidor en modo simulado estableciendo `MOCK_LLM=1` (`$env:MOCK_LLM="1"` en PowerShell o `MOCK_LLM=1 uvicorn server:app --port 8000` en Bash).

---

### Paso B: Ejecutar el Cliente del Estudiante (`cliente_estudiante/`)

Abre una **segunda terminal** en `grand_prix/cliente_estudiante/`:

```bash
cd grand_prix/cliente_estudiante
pip install -r requirements.txt
cp .env.example .env   # (en Windows PowerShell: Copy-Item .env.example .env)
python juego_pistas.py
```

#### Flujo recomendado para el estudiante
1. Edita `cliente_estudiante/.env` con tu `STUDENT_ID` y `STUDENT_NAME`.
2. **[1] Grabar Jugadas (Pistas A1, A2, A3):** Conduce con flechas/WASD (pulsa `N` para cambiar entre `A1`, `A2`, `A3` y `ESC` para guardar en `data/mis_jugadas.json`).
   *(Ya viene pre-cargado un archivo semilla en `data/mis_jugadas.json` con 120 ejemplos si deseas probar de inmediato; puedes limpiarlo con la tecla `C`).*
3. Edita `cliente_estudiante/config_estudiante.py` para elegir `ESTRATEGIA = "ICL"` o `"LORA"` y ajustar tus hiperparámetros.
4. Si elegiste `"LORA"`, ejecuta **[2] Entrenar Piloto**.
5. Ejecuta **[3] Modo Autónomo - Prueba (Pistas A)** para depurar antes de competir.
6. Ejecuta **[4] Evaluación Oficial (Pistas B1, B2, B3)** para enviar tu puntaje oficial a `/ranking`.

También puedes lanzar modos sin ventana gráfica desde CLI:
```bash
python juego_pistas.py --modo 2 --headless   # Entrenar LoRA
python juego_pistas.py --modo 4 --headless   # Evaluación oficial Pistas B
```

---

## 3. Despliegue en Clase: Google Colab Pro (A100 40 GB) + Cloudflare Tunnel

Para atender hasta **12 estudiantes en paralelo** en clase sin cambiar una sola línea de código del backend:

### Celda 1: Clonar o subir `grand_prix/backend_server` e instalar dependencias
```python
# En Google Colab Pro (Entorno -> NVIDIA A100 o L4)
!nvidia-smi
!pip -q install -r /content/grand_prix/backend_server/requirements.txt

# Instalar cloudflared para túnel público sin cuenta ni contraseña
!wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /usr/local/bin/cloudflared
```

*(Si subes la carpeta comprimida en `.zip` a Colab, descomprímela en `/content/grand_prix` antes de correr la celda siguiente).*

### Celda 2: Iniciar FastAPI (`server.py`) y publicar URL `trycloudflare.com`
```python
import os, re, subprocess, time, urllib.request

os.chdir("/content/grand_prix/backend_server")

# Asegurar .env activo
if not os.path.exists(".env") and os.path.exists(".env.example"):
    import shutil
    shutil.copy(".env.example", ".env")

# 1. Lanzar uvicorn en segundo plano
server_log = open("/content/uvicorn.log", "w")
server_proc = subprocess.Popen(
    ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"],
    stdout=server_log,
    stderr=subprocess.STDOUT,
)

# Esperar a que /health responda (carga Qwen2.5-1.5B-Instruct en GPU)
print("Cargando Qwen2.5-1.5B-Instruct en GPU...")
for _ in range(90):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as r:
            if r.status == 200:
                print("Backend listo:", r.read().decode())
                break
    except Exception:
        time.sleep(2)

# 2. Abrir túnel Cloudflare
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
print("  URL PÚBLICA PARA LOS ESTUDIANTES (pegar en SERVER_URL del .env):")
print(f"  SERVER_URL={public_url}")
print(f"  RANKING EN VIVO (proyector): {public_url}/ranking")
print("=" * 68)
```

> **Importante:** Usa siempre `--workers 1` en `uvicorn` para mantener una **única copia del modelo en VRAM** compartida y protegida por `gpu_lock`.

### Paso para los 12 estudiantes en clase
Cada estudiante solo cambia dos líneas en su `cliente_estudiante/.env`:
```env
SERVER_URL=https://tu-subdominio.trycloudflare.com
STUDENT_ID=estudiante_07
STUDENT_NAME=Maria_Perez
```
Y ejecuta `python juego_pistas.py` en su computador personal.

---

## 4. Endpoints del Backend (`backend_server/server.py`)

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado de GPU, VRAM libre, modelo base cargado y límites del `.env` |
| `POST` | `/entrenar_lora` | Valida límites (`MAX_LORA_R`, `MAX_LORA_EPOCHS`) y entrena adaptador LoRA en `adapters_cache/{student_id}` |
| `POST` | `/decidir_accion` | Ejecuta ICL (`knn`/`random`) o LoRA (`set_adapter`) y devuelve `{"accion": int, "latencia_gpu_ms": float}` |
| `POST` | `/registrar_puntaje` | Calcula puntaje oficial de Pistas `B1, B2, B3` penalizando latencia y actualiza `ranking.json` |
| `GET` | `/ranking` | Tabla HTML en vivo con auto-refresco cada 5 segundos |
| `GET` | `/ranking/json` | Tabla de posiciones en formato JSON |
