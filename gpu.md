# Guía de GPUs en Google Colab Pro (Top, Características y Tiempo de Uso)

La suscripción de **Google Colab Pro** incluye **100 Unidades de Cómputo (CUs) al mes** (aprox. \$9.99 USD/mes).  
En lugar de cobrar una tarifa fija por hora, Colab descuenta CUs según la GPU asignada, la RAM del sistema (*High-RAM* consume un poco más) y la demanda del servidor.

> **Nota:** Si se te acaban las 100 CUs antes de fin de mes, Colab Pro no te bloquea del todo: pasas a la cola estándar (similar al nivel gratuito con T4) o puedes recargar paquetes *Pay-As-You-Go*.

---

## Top de GPUs en Colab Pro (de mayor a menor potencia)

| Puesto | GPU | Arquitectura | VRAM | Ancho de Banda | Pico Tensor Core (BF16 / FP16 denso) | Consumo aprox. (CUs / hora) | **Tiempo al mes (100 CUs)** | **Clases de 1.5 h al mes** | Acceso típico |
|---:|---|---|---|---|---|---|---|---|---|
| **1** | **NVIDIA H100** | Hopper | **80 GB HBM3** | 3.35 TB/s | **1,979 TFLOPS** (BF16/FP16) | **~17 – 20 CUs/h** | **~5 – 6 horas / mes** | ~3 – 4 clases | Principalmente Pro+ / alta demanda |
| **2** | **NVIDIA A100** | Ampere | **40 GB / 80 GB HBM2e** | 1.6 – 2.0 TB/s | **312 TFLOPS** (BF16/FP16) | **~10.5 – 15 CUs/h** | **~7 – 10 horas / mes** | ~4 – 6 clases | Colab Pro / Pro+ |
| **3** | **NVIDIA L4** | Ada Lovelace | **24 GB GDDR6** | 300 GB/s | **242 TFLOPS** (BF16/FP16) | **~3.0 – 4.8 CUs/h** | **~20 – 33 horas / mes** | ~13 – 22 clases | Colab Pro / Pro+ |
| **4** | **NVIDIA T4** | Turing | **16 GB GDDR6** | 320 GB/s | **65 TFLOPS** (FP16; **sin** BF16 nativo) | **~1.1 – 1.8 CUs/h** | **~55 – 90 horas / mes** | ~36 – 60 clases | Gratis / Pro / Pro+ |

*(Las cifras de TFLOPS corresponden a cómputo denso; con esparsidad estructurada 2:4 pueden duplicarse).*

---

## Detalle de cada GPU: cuándo usarla y qué te conviene

### 1. NVIDIA H100 (80 GB HBM3) — *La más rápida*
- **Para qué sirve:** Entrenar o hacer *Full Fine-Tuning* de LLMs grandes (7B–32B), lotes enormes y contextos largos con **FlashAttention** y **BF16/FP8** nativo.
- **Lo malo en Colab Pro:** Suele estar restringida a planes superiores (Colab Pro+) o sujeta a cola estricta, y consume tus 100 CUs en apenas **5 a 6 horas**.

### 2. NVIDIA A100 (40 GB u 80 GB HBM2e) — *La reina de entrenamiento pesado en Pro*
- **Para qué sirve:** *Full Fine-Tuning* de modelos de 3B–7B, QLoRA de modelos 14B–32B, o experimentos donde necesites altísimo ancho de banda de memoria (`1.6–2.0 TB/s`).
- **Duración con tus 100 CUs:** **~7 a 10 horas** continuas.
- **Cuándo elegirla:** Solo cuando una L4 de 24 GB se quede corta de VRAM o necesites terminar un entrenamiento grande rápido.

### 3. NVIDIA L4 (24 GB GDDR6) — *El punto dulce (Recomendada para tu proyecto)*
- **Para qué sirve:** Tiene **24 GB de VRAM** y **242 TFLOPS en BF16 nativo** (casi 4× la potencia FP16 de una T4 y con arquitectura moderna *Ada Lovelace*), pero gasta menos de un tercio de CUs por hora que una A100.
- **Duración con tus 100 CUs:** **~20 a 33 horas** al mes (te alcanza para **13 a 20 sesiones de clase de 1.5 horas**!).
- **Por qué es ideal para `option1/` (`Qwen2.5-0.5B`):**
  - `Qwen2.5-0.5B` en BF16 pesa ~1 GB; incluso haciendo **Full Fine-Tuning** (~8–12 GB) sobra más de la mitad de la VRAM.
  - Además, en **GCP (`g2-standard-4`)** usarás exactamente la misma **NVIDIA L4**, así que los tiempos que midas en Colab Pro serán casi idénticos a los de tu clase en vivo.

### 4. NVIDIA T4 (16 GB GDDR6) — *La más económica para pruebas largas*
- **Para qué sirve:** Inferencia, *Prompting (ICL)* y **LoRA / QLoRA** de modelos pequeños (`0.5B` a `3B` en FP16 o 4 bits).
- **Limitación importante:** No tiene soporte nativo para `bfloat16` (usa `float16`), y es bastante más lenta que la L4 en entrenamiento.
- **Duración con tus 100 CUs:** **~55 a 90 horas** al mes.

---

## Recomendación rápida para probar `option1/`
1. En Colab Pro selecciona **L4 GPU** (Entorno de ejecución $\rightarrow$ Cambiar tipo de entorno de ejecución $\rightarrow$ **L4 GPU**).
2. Desactiva *High-RAM* si no lo necesitas para ahorrar CUs.
3. **Recuerda desconectar el entorno de ejecución** (*Entorno de ejecución $\rightarrow$ Desconectar y eliminar*) apenas termines tus pruebas: Colab sigue descontando CUs mientras la sesión siga viva aunque no estés ejecutando celdas.
