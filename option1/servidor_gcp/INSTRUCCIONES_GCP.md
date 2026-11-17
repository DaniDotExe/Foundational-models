# Guía Rápida para Desplegar el Servidor en GCP (12 Estudiantes · 1.5 Horas)

## 1. Crear la VM en Google Cloud (Compute Engine)
- **Tipo de máquina:** `g2-standard-4` (incluye **1× GPU NVIDIA L4 de 24 GB VRAM**, ~$0.75 USD/h) o `n1-standard-4` con **1× NVIDIA T4**.
- **Imagen (Boot Disk):** *Deep Learning VM with CUDA 12 / PyTorch* (Ubuntu 22.04, 50 GB SSD).
- **Regla de Firewall:** Permitir tráfico TCP en el puerto `8000` (o exponerlo rápido con `cloudflared tunnel --url http://localhost:8000` para tener HTTPS sin abrir puertos).

## 2. Arrancar el servidor en la VM (3 comandos)
```bash
pip install -r requirements.txt
export FM_MODEL_ID="Qwen/Qwen2.5-0.5B"
export RUN_REAL_GPU="1"
uvicorn server:app --host 0.0.0.0 --port 8000
```

## 3. Qué necesitas darle a tus 12 estudiantes al empezar la clase
1. La **URL/IP** de tu VM (ej. `http://34.xxx.xxx.xxx:8000` o la URL de `cloudflared`).
2. Su **`TEAM_TOKEN`** único (`token_estudiante_01` hasta `token_estudiante_12`).
3. La carpeta del estudiante (`config.py`, `entrenar_y_evaluar.py`, `simulador_y_arena.html` y `datos/`).
