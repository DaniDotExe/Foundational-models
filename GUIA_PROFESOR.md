# Guía Paso a Paso para el Profesor — UIS Grand Prix Foundationals Models

Guía práctica para dirigir la sesión (~3.5 horas) comparando **ICL**, **LoRA (PEFT)** y **LoRA + ICL** sobre **`Qwen/Qwen2.5-1.5B-Instruct`** con 12 estudiantes en paralelo.

---

## 1. Antes de que lleguen los estudiantes (5–10 min)

1. **Abre Colab (GPU L4)** con [colab_servidor.ipynb](backend_server/colab_servidor.ipynb).
2. *(Opcional si corriste pruebas antes)* Limpia los datos de prueba en una celda de Colab:
   ```python
   !rm -rf /content/Foundational-models/backend_server/ranking.json \
           /content/Foundational-models/backend_server/trayectorias_b.json \
           /content/Foundational-models/backend_server/adapters_cache/*
   ```
3. Ejecuta **Celda 1** (clona/actualiza el repo, quita `torchao` e instala dependencias) y luego **Celda 2** (inicia `uvicorn` + túnel `trycloudflare.com` + monitor en vivo).
4. Copia la URL pública que imprime la Celda 2:
   - `SERVER_URL=https://xxxx.trycloudflare.com` (compártela a tus estudiantes).
   - Abre en tu navegador y proyecta en el salón: **`https://xxxx.trycloudflare.com/ranking`** (se auto-refresca cada 5 s).
5. Deja la pestaña/celda de Colab abierta (irá imprimiendo `[ENTRENANDO]`, `[LORA LISTO]` y `[SIMULANDO]` en tiempo real).

---

## 2. Arranque de los estudiantes (10 min)

Pídeles que en su PC ejecuten:

```bat
git clone https://github.com/DaniDotExe/Foundational-models.git
cd Foundational-models
pip install -r participante/requirements.txt
```

Luego en `participante/.env`:
1. Peguen la URL que les diste en `SERVER_URL=https://xxxx.trycloudflare.com` (sin `/` al final).
2. Abran el simulador:
   ```bat
   python participante/juego_pistas.py
   ```
3. **MUY IMPORTANTE:** apenas abra el menú, cada alumno debe pulsar **`Editar Nombre`** y escribir **su nombre real** (si no lo cambian, varios quedarían como `Dani` y se sobrescribirían sus adaptadores LoRA y su puesto en `/ranking`).

---

## 3. Qué explicar y cómo guiar cada bloque de la clase

### Bloque 1 — El problema y el Dataset (`[G] Grabar Jugadas y Ver Frames`)
- **Qué ven en pantalla (Concepto de Estado -> Acción):**
  - **Entradas al modelo:** 5 rayos de distancia (`izq`, `diag_izq`, `frente`, `diag_der`, `der`), distancia y ángulo al próximo checkpoint (`cp_dist`, `cp_ang`), `velocidad` y el `Historial` de las últimas acciones.
  - **Salida (1 solo token):** `0 = Recto`, `1 = Izquierda`, `2 = Derecha`, `3 = Frenar`.
- **Actividad:**
  - Conducir en **`P1`, `P2`, `P3`** grabando jugadas limpias.
  - Entrar a **`Ver Frames`** para revisar qué grabaron, usar la cuadrícula (`3×3`) o borrar frames malos (`Supr`/`Backspace`, `Ctrl+Z` deshace).
  - *Nota práctica:* si el botón `Ver Frames` se pone **rojo** (`> 300`), conviene limpiar frames repetidos o innecesarios (el servidor usa máximo **300**).

---

### Bloque 2 — Experimento 1: **`ICL` (In-Context Learning)**
- **Idea clave a explicar:** el modelo base **NO cambia sus pesos**. Aprende únicamente de los ejemplos que le metemos en el prompt en cada frame.
- **Qué deben probar dentro de la tarjeta `ICL`:**
  - Por defecto viene en nivel medio-bajo (`k_shots = 4`, `seleccion = RANDOM`, `temperature = 0.2`, `precision = 20 bins`).
  - Hacer una primera simulación en **`Simular Prueba (P1-P3)`** (en cola en segundo plano) para ver cómo falla con pocos ejemplos aleatorios.
  - Luego mejorar paso a paso usando los botones **`[i]`**:
    1. Cambiar `seleccion` de **`RANDOM` -> `KNN`** (buscar vecinos más cercanos en vez de azar).
    2. Subir `k_shots` (ej. `8..15`) y poner `temperature = 0.0`.
    3. Ajustar `ventana_historia` (`4..5`) y `precision_sensores` (`50`).
  - **Lección pedagógica:** `ICL` itera al instante (0 s de entrenamiento), pero entre más `k_shots` metes, **más largo es el prompt y más sube la latencia** (lo verán en el Ranking de Velocidad).

---

### Bloque 3 — Experimento 2: **`LORA` (Fine-Tuning con PEFT)**
- **Idea clave a explicar:** el modelo base (`Qwen2.5-1.5B`) sigue congelado para todos, pero cada estudiante entrena unas **matrices pequeñas de bajo rango (LoRA)** propias. Al simular en `LORA`, el modelo corre **zero-shot** (0 ejemplos en el prompt).
- **Qué deben probar dentro de la tarjeta `LORA`:**
  - Usar los botones **`[i]`** para entender cada parámetro y sus restricciones (`num_ejemplos` 1..300, `lora_r` 1..32 potencia de 2, `lora_alpha`, `epochs` 1..15, `learning_rate`, `Solo Atención` vs `Atención + MLP`).
  - Entrenar una primera versión (`v1_...` con el baseline `r=4, epochs=3, 60 ejemplos`) y ver la barra de progreso en vivo debajo de los botones.
  - Entrenar una segunda/tercera versión (`v2_...`, `v3_...`) subiendo `num_ejemplos` (`150–250`), `lora_r` (`16`), `lora_alpha` (`32`), `epochs` (`6–8`) y probando `Atención + MLP`.
  - Comparar sus 3 versiones directamente en **`Simular Prueba (P1-P3)`** usando el selector **`Modelo LoRA a usar`**.
  - **Lección pedagógica:** `LORA` invierte unos segundos en entrenar, pero luego responde mucho más rápido (prompt corto) y generaliza mejor en pistas nuevas.

---

### Bloque 4 — Experimento 3: **`LORA + ICL` (Híbrido)**
- **Idea clave a explicar:** combina un adaptador **LoRA** ya entrenado **+** `k_shots` ejemplos en el prompt.
- **Pregunta para el grupo:** ¿darle ejemplos en contexto a un modelo ya entrenado con LoRA mejora el desempeño en curvas difíciles, o solo agrega ruido y más latencia?
- En **`Simular Prueba (P1-P3)`**, verán etiquetado claramente `LORA (vX) + ICL (k=Y)` para comparar contra `LORA` puro y `ICL` puro.

---

### Bloque 5 — Evaluación Oficial (`B1, B2, B3`), Ranking F1 y Carrera Final
1. Pídeles que ejecuten **`Evaluacion Oficial (B1-B3)`** con su mejor configuración (pueden registrar una participación con **`ICL`**, una con **`LORA`** y una con **`LORA_ICL`** para compararlas en la tabla).
2. Proyecta **`/ranking`** y recorre con ellos las **5 pestañas**:
   - **🏆 Campeonato General F1:** `0.55·Carrera + 0.20·Velocidad + 0.15·Few-Shot + 0.10·Limpieza`.
   - **🏁 Carrera (55%):** quién completa más pistas ocultas (`B1`, `B2`, `B3`), `%` promedio y tiempo físico.
   - **⚡ Velocidad (20%):** latencia GPU promedio (`ms/frame`).
   - **🧬 Few-Shot (15%):** `% progreso / #ejemplos` (premia lograr más con menos datos).
   - **🛡️ Limpieza (10%):** menor número de choques en `B1–B3`.
3. **Cierre en tu PC (`[C] Ver Carrera Final (B1-B3)`):**
   - Pulsa **`[C]`** en el menú principal e ingresa tu clave docente (**`dani`**).
   - Proyecta **`Modo Fantasma (Transparente)`** en `B1`, `B2` y `B3` para ver a todos los carros competir superpuestos.
   - Luego pulsa **`M`** para activar el **`MODO LEGENDARIO (Choques ON)`** y ver qué pasa cuando todos los pilotos autónomos colisionan entre sí en la pista.
