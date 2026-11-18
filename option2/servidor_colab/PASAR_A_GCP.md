# Cómo pasar de Google Colab Pro a GCP (Producción en Clase)

El mismo código de `server.py` y `arena_pixelart.html` funciona sin cambios en ambos entornos. Cuando termines de probar en Colab Pro y quieras levantarlo en GCP para los 12 estudiantes:

1. **Crear VM en GCP:**
   - Máquina recomendada: `g2-standard-8` (**1× NVIDIA L4 24 GB**, 32 GB RAM, ~\$1.00 USD/h).
   - Imagen: *Deep Learning VM with CUDA 12 / PyTorch (Ubuntu 22.04)*.

2. **Copiar la carpeta `option2/` a la VM e instalar dependencias:**
   ```bash
   pip install fastapi uvicorn transformers peft accelerate
   ```

3. **Arrancar el servidor con los mismos parámetros:**
   ```bash
   export RUN_REAL_GPU="1"
   export COOLDOWN_ENTRENAMIENTO_S="180"   # 1 entrenamiento cada 3 min por equipo
   export ARENA_SAVE_DIR="./estado_arena"
   uvicorn option2.servidor_colab.server:app --host 0.0.0.0 --port 8000
   ```

4. **Exponer por HTTPS sin configurar firewall (opcional, igual que en Colab):**
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
   Comparte esa URL con los 12 estudiantes; al abrirla verán directamente `arena_pixelart.html` conectada a tu GPU de GCP.
