# Option 2: Juegos del Hambre de Modelos Fundacionales

Diseño base del reto multi-modelo con presupuesto fijo de FLOPs, simulador local contra agentes imaginarios y Arena PvP final en Pixel Art.

---

## 1. Los 3 Modelos Base e Implicaciones

Todos comparten la misma familia (**Qwen2.5-Instruct**) para mantener el mismo tokenizador y formato de chat, variando únicamente la escala ($N$) dentro de un límite fijo de **$1.0 \times 10^{15}$ FLOPs (100 créditos)** por entrenamiento:

| Modelo | Parámetros ($N$) | VRAM (`bf16`) | Estrategias viables bajo el límite de FLOPs |
| :--- | :--- | :--- | :--- |
| `Qwen/Qwen2.5-0.5B-Instruct` | ~0.49B | ~1.0 GB | `Full-FT`, `LoRA` pesado (muchos datos $D$) o `ICL` |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~1.54B | ~3.0 GB | `LoRA` ligero/medio o `ICL + CoT` |
| `Qwen/Qwen2.5-3B-Instruct` | ~3.09B | ~6.0 GB | Principalmente `ICL + CoT` (o `LoRA` muy corto) |

### Implicaciones técnicas (12 estudiantes)
1. **VRAM:** Los 3 modelos cargados juntos en inferencia ocupan **~10 GB**. En una **NVIDIA L4 (24 GB)** o **A100 (40 GB)** caben todos, pero conviene restringir `Full-FT` únicamente al `0.5B` (o descargar temporalmente el `3B` durante el entrenamiento) para evitar OOM.
2. **Trade-off pedagógico (Chinchilla):** Quien elige el modelo más grande ($3\text{B}$) casi no tiene FLOPs para entrenar pesos; quien elige el chico ($0.5\text{B}$) puede entrenarlo a fondo con datos curados.
3. **Arena PvP:** En cada turno se agrupan los tributos vivos por tamaño de modelo (hasta 3 mini-batches por turno). Duración estimada de 25 turnos: **~2.5–4.5 min en L4** o **~1–2 min en A100**.

---

## 2. Stats Entrenables (Habilidades)

Cada tributo adapta su modelo para 4 tareas concretas (el estudiante decide la vía —`ICL`, `LoRA` o `Full-FT`— y la curación de datos para cada una):

1. **Comer (Forrajeo):** Distinguir comida segura de bayas venenosas o trampas a partir de descripciones con ruido. Recupera **Hambre** y evita daño por veneno.
2. **Guerra (Combate):** Decidir `ATACAR`, `DEFENDER` o `EMBOSCAR` según vida, arma y estado del rival cercano.
3. **Correr (Evasión y Navegación):** Calcular hacia qué coordenada moverse para escapar de rivales fuertes o de la zona que se encoge (aquí brilla *Chain-of-Thought*).
4. **Sigilo / Astucia:** Detectar rastros de otros tributos y decidir si saquear suministros o esconderse.

---

## 3. Peligros y Bonus de la Arena

### Peligros
* **Hambre progresiva:** Cada turno consume energía; si llega a `0`, el tributo pierde vida cada turno.
* **Zona que se encoge (Niebla del Capitolio):** A partir del turno 6, el perímetro se cierra progresivamente e inflige daño por turno a quien quede fuera.
* **Comida envenenada / Trampas:** Castiga a modelos que no filtraron datos corruptos en *Comer*.
* **Eventos / Bestias:** Cada 5 turnos aparece un peligro en una zona aleatoria del mapa.

### Bonus
* **La Cornucopia (Turno 1):** Recursos de alto valor en el centro (armas `+20` ataque, raciones extra) a cambio de riesgo de combate temprano.
* **Cajas de Patrocinadores:** Suministros que caen en el mapa (curan vida o recargan estamina).
* **Bono de Eficiencia:** Sobrar créditos del presupuesto de 100 otorga un pequeño bono inicial (ej. ración extra o +10 estamina).

---

## 4. Despliegue en Google Colab Pro (`.ipynb`)

Es totalmente viable montar el servidor en un único notebook `.ipynb` de Colab Pro (`L4` o `A100`):
* **Exposición a los 12 estudiantes:** `uvicorn` (FastAPI) corriendo en segundo plano + túnel `cloudflared` (URL HTTPS pública instantánea).
* **Persistencia:** Montar Google Drive para guardar los adaptadores LoRA / checkpoints de los 12 equipos y el `arena_replay.json`.
* **Visualización:** El notebook genera/sirve el `arena_pixelart.html` en `<canvas>` 2D para proyectar en clase sin consumir GPU en gráficos.
