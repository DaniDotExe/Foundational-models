# Guía del Profesor: Modelos Fundacionales

> **Enfoque de este documento:** Este archivo concentra todo el respaldo pedagógico, cronograma detallado, preguntas guía, analogías, guiones de demostración y profundidad científica/matemática para que el docente domine la sesión.  
> **Presentación proyectada para el estudiante:** [FM.html](file:///d:/Software/Foundational%20models/FM.html) (17 diapositivas minimalistas, visuales y libres de tiempos o jerga innecesaria).

---

## 1. Mapa General y Cronograma (90 Minutos)

| Bloque | Tiempo | Diapositivas | Objetivo Pedagógico | Analogía / Dinámica Clave |
| :--- | :--- | :--- | :--- | :--- |
| **Portada** | `0 min` *(inicio)* | `01` | Presentar el recorrido de la clase | Mapa visual de los 6 bloques |
| **Bloque 1: El Gran Cambio de Paradigma** | `0 – 15 min` | `02` – `04` | Entender qué es un modelo fundacional y por qué dejó obsoleta a la IA tradicional | Pregunta gancho inicial · Bloque de mármol · Chequeo rápido (ChatGPT legal) |
| **Bloque 2: La Receta de Construcción** | `15 – 35 min` | `05` – `07` | Desmitificar el tamaño y explicar el balance matemático entre datos y parámetros | Genio con 2 libros vs. profesional con biblioteca · Votación Modelo A vs. B |
| **Bloque 3: Habilidades Emergentes** | `35 – 50 min` | `08` – `09` | Explicar cómo el crecimiento cuantitativo produce capacidades cualitativas nuevas | Agua calentándose hasta 100 °C → vapor · Discusión sobre imprevisibilidad |
| **Bloque 4: El Usuario al Mando** | `50 – 68 min` | `10` – `12` | Explicar cómo se adapta y guía un modelo sin programar ni reentrenar | Demostración en vivo (*few-shot* + paso a paso) · Actor con libreto 5 min antes |
| **Bloque 5: La Realidad Práctica** | `68 – 82 min` | `13` – `15` | Comprender riesgos, límites de alineación y propagación de fallos | Máscara amable sobre modelo crudo · Punto Único de Falla · Arquitectura segura |
| **Bloque 6: Síntesis, Preguntas y Cierre** | `82 – 90 min` | `16` – `17` | Consolidar el mapa mental completo y abrir el debate final | Pregunta de salida para la clase |

---

## 2. Corpus Científico de Respaldo (Para Preguntas Profundas)

Todas las afirmaciones conceptuales y cifras técnicas provienen de las 8 obras primarias en `references/`:
1. **Bommasani, R., Liang, P., et al. (2021).** *On the Opportunities and Risks of Foundation Models*. CRFM, Stanford University. *(Origen del término, homogeneización, Punto Único de Falla, riesgo sistémico).*
2. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. Google AI Language. *(Autosupervisión con huecos `[MASK]` y transferencia multi-tarea).*
3. **Kaplan, J., McCandlish, S., et al. (2020).** *Scaling Laws for Neural Language Models*. OpenAI. *(Leyes de potencias y relación $C \approx 6ND$).*
4. **Brown, T. B., et al. (2020).** *Language Models are Few-Shot Learners (GPT-3)*. NeurIPS / OpenAI. *(Aprendizaje en contexto sin modificar pesos: zero-shot, one-shot, few-shot y límites de escalado puro).*
5. **Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022).** *Training Compute-Optimal Large Language Models (Chinchilla)*. NeurIPS / DeepMind. *(Escalamiento equilibrado 1:1 y regla de ~20 tokens por parámetro).*
6. **Wei, J., Tay, Y., Bommasani, R., et al. (2022).** *Emergent Abilities of Large Language Models*. TMLR / Google Research. *(Transiciones de fase no lineales y Chain-of-Thought).*
7. **Chen, P.-Y., & Liu, S. (2024).** *Introduction to Foundation Models*. Springer Nature. *(Taxonomía FM vs. LLM vs. GenAI, ciclo de vida y alineación).*
8. **Huyen, C. (2025).** *AI Engineering: Building Applications with Foundation Models*. O'Reilly Media. *(Economía de inferencia, Llama 3, post-entrenamiento, seguridad, RAG y evaluación práctica).*

> **Controles de [FM.html](file:///d:/Software/Foundational%20models/FM.html) durante la clase:**  
> `→` o `Espacio` revela paso a paso las animaciones (útil para pausar antes de mostrar respuestas de preguntas/ejercicios); `←` retrocede; `G` abre el índice visual de las 17 diapositivas; `A` muestra todo sin animaciones; `F` pantalla completa; `P` imprime o guarda en PDF.

---

# BLOQUE 1: El Gran Cambio de Paradigma (0 – 15 min)

> **Objetivo:** Que los estudiantes comprendan qué es un modelo fundacional y por qué dejó obsoleta a la IA tradicional.

## Diapositiva 01 — Portada
- **Tiempo:** `0 min` (arranque mientras se acomoda el grupo).
- **Nota docente:** Presenta la sesión no como un catálogo de herramientas de moda, sino como la historia de *cómo funciona por dentro* la IA moderna, desde su base matemática hasta su uso diario.

## Diapositiva 02 — ¿Un programa distinto para cada tarea? *(0 – 5 min | El gancho inicial)*
- **Tiempo:** `0 – 5 min`
- **Secuencia de clics en pantalla:**
  - *Inicio de diapositiva:* Solo se muestra la pregunta inicial en pantalla. **No hagas clic todavía.**
  - *Clics 1, 2 y 3:* Aparecen sucesivamente **Modelo A** (Traducir), **Modelo B** (Resumir contratos) y **Modelo C** (Escribir código).
  - *Clic 4:* Aparece la síntesis inferior sobre el esquema tradicional ("uno por tarea").
- **Guion y dinámica docente:**
  - Lanza la pregunta al grupo: *«Si quisieran un programa que traduzca textos, otro que resuma contratos y otro que escriba código, ¿cuántos programas necesitarían programar?»*
  - Deja que respondan intuitivamente ("tres programas distintos").
  - Haz clic 1, 2 y 3 para explicar que así funcionaba la IA tradicional (**esquema *one-for-one***): un modelo exclusivo, diseñado y entrenado desde cero para cada problema específico, sin transferir aprendizaje de uno a otro.
- **Respaldo técnico para el profesor (no proyectar):**
  - Según **Chen & Liu (2024, §1.1, p. 14)**, antes coexistían arquitecturas aisladas (CNN para visión, LSTM/clasificadores dedicados por tarea).
  - El gran cuello de botella era el **etiquetado manual supervisado**. **Huyen (2025, pp. 45–46)** calcula que, a \$0.05 por etiqueta, etiquetar 1 millón de imágenes de 1,000 categorías (ImageNet) costó \$50,000–\$100,000; escalar a 1 millón de conceptos costaría **\$50 millones**, volviéndose inviable en medicina o derecho.

## Diapositiva 03 — La Nueva Arquitectura: Preentrenamiento + Adaptación *(5 – 12 min)*
- **Tiempo:** `5 – 12 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Tarjeta izquierda — Autosupervisión (ejemplos GPT y BERT).
  - *Clic 2:* Tarjeta derecha — Bloque de Mármol (preentrenamiento masivo).
  - *Clic 3:* Ramificación con el cincel — Estatua Ecuestre, Columna y Vasija (adaptación).
- **Guion y dinámica docente:**
  1. Introduce el concepto de **Modelo Fundacional** (*Foundation Model*, **Bommasani et al., 2021; Chen & Liu, 2024**). Explica por qué los investigadores de Stanford (Percy Liang, Rishi Bommasani y >100 coautores en agosto de 2021) eligieron la palabra *"cimiento"*: es una base deliberadamente incompleta sobre la que se construyen decenas de aplicaciones distintas.
  2. Explica la **autosupervisión (Clic 1):** ya no hay personas etiquetando datos a mano. El modelo genera sus propios ejercicios de práctica a partir de texto crudo:
     - Estilo **GPT:** predecir la palabra que sigue (*"El contrato establece que las partes → [acuerdan]"*).
     - Estilo **BERT:** rellenar palabras ocultas en una oración (*"La cláusula de [hueco] aplica desde hoy"*).
  3. Presenta **la analogía central (Clics 2 y 3):**
     - **El bloque de mármol:** el preentrenamiento extrae de la cantera de internet un bloque masivo y sólido de mármol. Todavía no es una obra usable, pero contiene toda la estructura.
     - **El cincel (adaptación):** mediante *fine-tuning* o *prompting*, un ingeniero da unos cuantos toques de cincel para esculpir el mismo bloque en estatua ecuestre (asistente de chat), columna (resumen legal) o vasija (programación).
- **Respaldo técnico para el profesor (distinciones por si preguntan):**
  - *¿Cuándo ocurrió el punto de quiebre?* Con **BERT (Devlin et al., 2019, p. 5)**: tras preentrenarse en BooksCorpus (800M palabras) + Wikipedia (2,500M palabras), superó el estado del arte en **11 tareas distintas** añadiendo solo una capa final y ajustándose en menos de 1 hora en 1 Cloud TPU.
  - *¿Todo Modelo Fundacional es un LLM o generativo?* No (**Chen & Liu, 2024, Fig. 1.2, pp. 15–16; Huyen, 2025, pp. 49–52, 147**). Existen FMs de visión (**DINOv2**), biología (**AlphaFold**, ~100k proteínas) y multimodales (**CLIP**, 400M pares imagen-texto), y muchos son *encoders* no generativos (solo producen *embeddings*).
  - *¿Por qué existen modelos distintos para texto, imagen y video?* Comprender entradas converge en un mismo espacio multimodal, pero generar píxeles o video a 60 fps requiere modelos de difusión continuos en vez de predicción discreta token a token, además de diferencias enormes de VRAM (2 GB/1B params en FP16) y redundancia temporal frente a la alta densidad semántica del texto (100 tokens ≈ 75 palabras, **Huyen, 2025, pp. 39, 50–52, 166, 869**).

## Diapositiva 04 — Chequeo Rápido de Comprensión *(12 – 15 min)*
- **Tiempo:** `12 – 15 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Aparece únicamente la pregunta para reflexionar en clase. **Pausa aquí para escuchar al grupo.**
  - *Clic 2:* Aparece a la izquierda el mito tachado y a la derecha la explicación real en 3 pasos.
- **Guion y dinámica docente:**
  - Lanza la **pregunta trampa**: *«Entonces, ¿ChatGPT viene programado de fábrica con todas las respuestas a preguntas legales?»*
  - Escucha 2 o 3 opiniones antes de hacer Clic 2.
  - **Cómo guiar la respuesta (Clic 2):** Explica que **no** contiene una base de datos de preguntas y respuestas preescritas (`if pregunta == X return Y`). Lo que tiene es una **comprensión general y estadística de las estructuras del lenguaje y conceptos** (preentrenamiento) que luego se **adaptó** para entender órdenes y redactar una respuesta nueva palabra por palabra al momento.

---

# BLOQUE 2: La Receta de Construcción: Escalamiento y Cómputo Óptimo (15 – 35 min)

> **Objetivo:** Desmitificar el tamaño de los modelos y explicar cómo la ciencia descubrió el balance matemático entre datos y parámetros.

## Diapositiva 05 — La Receta Matemática: Leyes de Escalamiento *(15 – 22 min)*
- **Tiempo:** `15 – 22 min`
- **Secuencia de clics en pantalla:**
  - *Base fija:* Curva limpia de Ley de Potencias ("Más escala = Menos error de forma predecible").
  - *Clics 1, 2 y 3:* Aparecen una a una las 3 variables del presupuesto: **Cómputo (`C`)**, **Parámetros (`N`)** y **Datos (`D`)**, más la regla de presupuesto compartido ($C \approx 6 \cdot N \cdot D$).
- **Guion y dinámica docente:**
  - Explica que en IA moderna el rendimiento no mejora por suerte ni por intuición artesanal: sigue una **curva matemática predecible** conocida como Ley de Potencias (*Power Laws*, **Kaplan et al., 2020**). Antes de gastar millones, los laboratorios pueden estimar cuánto bajará el error al invertir más recursos.
  - Desglosa las **3 variables del presupuesto**:
    1. **Cómputo (FLOPs):** Operaciones matemáticas totales ejecutadas durante todo el entrenamiento. *(Aclaración docente: no confundir FLOPs totales con FLOP/s, que mide la velocidad por segundo de una GPU).*
    2. **Parámetros ($N$):** Cantidad de pesos o "conexiones" ajustables dentro de la red neuronal.
    3. **Datos ($D$ en tokens):** Volumen total de texto/información que lee el modelo durante el entrenamiento (1 token $\approx 0.75$ palabras).
  - Señala la ecuación simplificada $C \approx 6 \cdot N \cdot D$: como el presupuesto de cómputo $C$ es dinero y electricidad finita, si gastas demasiado haciendo crecer $N$, te quedas sin presupuesto para alimentar suficientes datos $D$.
- **Respaldo técnico para el profesor (no proyectar salvo que pregunten):**
  - **Derivación de $6ND$ (Kaplan et al., 2020, Tabla 1, pp. 6–7):** Cada token que pasa por cada parámetro hace una multiplicación y una suma en el *forward pass* ($2N$ FLOPs/token) y el doble en el *backward pass* para calcular gradientes de activaciones y pesos ($4N$ FLOPs/token), sumando $6N$ FLOPs por token.
  - **Ejemplo real (Huyen, 2025, pp. 170–172):** Entrenar GPT-3 ($N=175\text{B}$, $D=300\text{B}$) consumió $3.14 \times 10^{23}$ FLOPs ($\approx \$4.14\text{M}$ USD en 256 GPUs H100).
  - **Leyes de Kaplan (pp. 3–5, 14–16):** $L(N) = (N_c/N)^{0.076}$, $L(D) = (D_c/D)^{0.095}$, $L(C_{\min}) = (C_c/C_{\min})^{0.050}$. Kaplan observó que la forma exacta de la red (profundidad vs. anchura) importaba poco frente a la escala total.

## Diapositiva 06 — La Corrección de Chinchilla: El Balance Óptimo *(22 – 30 min)*
- **Tiempo:** `22 – 30 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Panel izquierdo — Antes (modelos gigantes hambrientos de datos) vs. Descubrimiento de Chinchilla (DeepMind).
  - *Clic 2:* Panel derecho — La regla de **~20 tokens por parámetro** y la analogía visual (genio vs. profesional).
- **Guion y dinámica docente:**
  - Cuenta la **historia histórica:** entre 2020 y 2022, siguiendo la conclusión original de Kaplan ($N \propto C^{0.73}, D \propto C^{0.27}$), toda la industria creyó que ante más presupuesto había que inflar casi todo en tamaño de parámetros ($5\times$ más grande el modelo y solo $2\times$ más datos). Así nacieron gigantes como GPT-3 (175B), Gopher (280B) y MT-NLG (530B), todos leídos con apenas ~300B tokens: modelos gigantescos pero **hambrientos de datos (*undertrained*)**.
  - Introduce el hallazgo de **DeepMind / Chinchilla (Hoffmann et al., 2022; Huyen, 2025):** entrenando más de 400 modelos (y corrigiendo el ciclo fijo de *learning rate* usado por Kaplan), demostraron que para un **cómputo óptimo**, parámetros y datos deben duplicarse a la par ($N \propto C^{0.50}, D \propto C^{0.50}$): aproximadamente **~20 tokens de datos por cada parámetro**.
  - Usa **la analogía de la diapositiva:**
    - *Modelo antiguo gigante:* Es como un **genio con una capacidad cerebral enorme que en toda su vida solo leyó dos libros** (mucho músculo sin suficiente experiencia).
    - *Modelo Chinchilla:* Es un **profesional con buena capacidad que leyó una biblioteca entera**. Al usar el mismo presupuesto total de cómputo que Gopher, Chinchilla fue $4\times$ más pequeño (70B vs. 280B), leyó $4.6\times$ más datos (1.4T vs. 300B tokens) y lo superó ampliamente (67.6% vs. 60.0% en MMLU).
- **Respaldo técnico para el profesor:**
  - **Fórmula de Hoffmann et al. (2022, p. 5):** $\hat{L}(N, D) = E + A/N^\alpha + B/D^\beta$ ($E$ entropía irreducible del lenguaje, $A/N^\alpha$ error de capacidad finita, $B/D^\beta$ error por datos finitos).
  - **Error metodológico de Kaplan:** Kaplan usó un *cosine learning rate schedule* fijo de 130B tokens para todas las corridas, midiendo pérdidas intermedias antes de completar la bajada del *learning rate*, lo que penalizó injustamente a modelos pequeños entrenados con más datos (Hoffmann et al., p. 3).

## Diapositiva 07 — Ejercicio Mental con la Clase: ¿Cuál elegirías? *(30 – 35 min)*
- **Tiempo:** `30 – 35 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Muestra únicamente las dos opciones teóricas frente a frente (**Modelo A: 500B / 1T tokens** vs. **Modelo B: 70B / 2T tokens**) y la pregunta. **¡Detente aquí antes de hacer Clic 2!**
  - *Clic 2:* Revela el veredicto en verde (`Ganador: Modelo B`) y las 3 razones de eficiencia en producción.
- **Guion y dinámica docente:**
  - Pide a la clase que levante la mano o vote en el chat: *«Si ustedes fueran el equipo de ingeniería de una empresa y tuvieran que poner uno de estos dos modelos en una aplicación real para miles de usuarios, ¿cuál elegirían y por qué?»*
  - Escucha argumentos (muchos al inicio se dejan seducir por el número "500B", pensando que más grande siempre gana).
  - Haz Clic 2 para revelar y explicar la respuesta:
    1. **Más rápido (menor latencia):** El usuario espera menos milisegundos por cada palabra, porque la red recorre 7 veces menos parámetros en cada paso.
    2. **Más barato en producción (inferencia):** Un modelo se entrena *una vez*, pero se consulta *millones de veces*. El costo operativo en GPUs y memoria VRAM depende directamente de $N$ (parámetros), no de cuánto entrenó.
    3. **Mayor densidad de conocimiento:** El Modelo A apenas vio 2 tokens por parámetro (muy subentrenado); el Modelo B vio ~28.5 tokens por parámetro, empaquetando mucho más conocimiento útil en un cuerpo compacto.
- **Nota de contexto moderno (Huyen, 2025, pp. 165, 168, 176):** Hoy en día las empresas van incluso más allá de Chinchilla (*over-training*): Meta entrenó **Llama 3 (8B)** con **15 Trillion (billones) de tokens** (~1,875 tokens por parámetro, casi $100\times$ la regla de Chinchilla), logrando que un modelo de apenas 8B supere al antiguo Llama 2 de 70B y pueda correr en una sola GPU comercial económica.

---

# BLOQUE 3: La Sorpresa Científica: Habilidades Emergentes (35 – 50 min)

> **Objetivo:** Explicar cómo el crecimiento puramente cuantitativo produce capacidades cualitativas nuevas e imprevistas.

## Diapositiva 08 — «More is Different»: Habilidades Emergentes *(35 – 42 min)*
- **Tiempo:** `35 – 42 min`
- **Secuencia de clics en pantalla:**
  - *Base + Clic 1:* Gráfica simplificada de transición brusca ("Nivel azar → Umbral Crítico → Despegue no lineal").
  - *Clic 2:* Los 4 ejemplos reales evaluados (BIG-Bench).
- **Guion y dinámica docente:**
  - Contrasta con el bloque anterior: el error básico del modelo (*loss*) baja de forma suave, **pero** cuando evaluamos habilidades complejas en la práctica ocurre una sorpresa (**Wei et al., 2022**, inspirados en la frase *"More Is Different"* del físico Philip Anderson, 1972).
  - Una **habilidad emergente** es una capacidad que está prácticamente en **0% (o nivel de azar)** en modelos pequeños y medianos, pero al cruzar cierto **umbral crítico de escala**, se dispara de forma repentina y no lineal.
  - Recorre los 4 ejemplos evaluados en *BIG-Bench*:
    1. **Aritmética de múltiples dígitos:** Sumar/restar 3 dígitos aparece en torno a 13B parámetros / $2.3\times 10^{22}$ FLOPs en GPT-3.
    2. **Traducción de idiomas con pocos recursos:** Comprender lenguas poco frecuentes en internet sin entrenamiento específico.
    3. **Comprensión de ironía y doble sentido:** Detectar sarcasmo o contexto social sutil.
    4. **Razonamiento paso a paso (*Chain-of-Thought*):** Desglosar problemas lógicos en pasos intermedios (emerge alrededor de 68B–100B / $10^{23}$ FLOPs; en modelos muy pequeños incluso empeora la respuesta).
- **Matiz científico importante para el profesor (por si surge debate avanzado):**
  - *Wei et al. (2022, §5.1, p. 7)* y *Schaeffer et al. (citado en Chen & Liu, 2024, p. 17)* señalan que parte de esa "brusquedad" aparente depende de cómo medimos: en tareas todo-o-nada (*Exact Match*), si aciertas 4 de 5 pasos de una operación sacas 0%; cuando el modelo alcanza el umbral de acertar los 5 pasos seguidos, la métrica salta de 0% a 80% de golpe, aunque debajo su probabilidad interna (*cross-entropy*) haya ido mejorando suavemente.

## Diapositiva 09 — Agua y Vapor, Mito de la Calculadora y Discusión *(42 – 50 min)*
- **Tiempo:** `42 – 50 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Diagrama Agua (20 °C → 90 °C, cambio cuantitativo) vs. Vapor (100 °C, transición de fase cualitativa).
  - *Clic 2:* Aclaración del error común (no hay calculadora enchufada adentro).
  - *Clic 3:* Pregunta de discusión rápida para cerrar el bloque.
- **Guion y dinámica docente:**
  - **42 – 47 min | La analogía del agua y el vapor:** Explica la diferencia entre cambio cuantitativo y cualitativo. Si calientas agua de 20 °C a 90 °C, solo obtienes agua más caliente (más cantidad de lo mismo). Pero al llegar exactamente a **100 °C**, ocurre una **transición de fase**: el líquido se vuelve vapor con propiedades físicas totalmente distintas. Escalar redes neuronales produce un fenómeno análogo.
  - **Aclara el error común (Clic 2):** Muchos piensan que cuando un modelo aprende a multiplicar o programar es porque algún ingeniero "le enchufó una calculadora interna" o escribió reglas en `if/else`. **Falso:** nadie programó esas reglas; la capacidad surgió espontáneamente al reorganizarse las representaciones geométricas de la red para predecir mejor el lenguaje.
  - **47 – 50 min | Discusión rápida (Clic 3):** Pregunta al grupo: *«¿Por qué esto asombró y preocupó tanto a la comunidad científica?»*
    - **Respuesta esperada:** Porque rompe la capacidad de predicción segura. No puedes saber mirando un modelo mediano qué capacidades extraordinarias —**ni qué vulnerabilidades o comportamientos peligrosos**— aparecerán de golpe en la siguiente versión tras invertir cientos de millones de dólares en entrenarlo.

---

# BLOQUE 4: El Usuario al Mando: Aprendizaje en Contexto (50 – 68 min)

> **Objetivo:** Explicar cómo se usa y adapta un modelo sin necesidad de programar ni reentrenar sus pesos.

## Diapositiva 10 — Del Fine-Tuning al In-Context Learning *(50 – 58 min)*
- **Tiempo:** `50 – 58 min`
- **Secuencia de clics en pantalla:**
  - *Banner superior:* Contraste directo entre modificar pesos (antes) vs. pesos congelados (ahora).
  - *Clics 1, 2 y 3:* Aparecen **Zero-shot**, **One-shot** y **Few-shot** con sus ejemplos visuales claros.
- **Guion y dinámica docente:**
  - Recuerda a los estudiantes que antes (incluso con BERT en 2019), para enseñar una tarea nueva había que descargar el modelo, montar GPUs, calcular gradientes y modificar sus pesos numéricos (*Fine-Tuning*).
  - Con la llegada de la escala masiva en GPT-3 (**Brown et al., 2020, p. 2**), apareció el **Aprendizaje en Contexto (*In-Context Learning*)**: el modelo se vuelve un meta-aprendiz capaz de mantener **todos sus parámetros congelados** y aprender una tarea nueva al instante simplemente leyendo los ejemplos en tu mensaje de entrada (*prompt*).
  - Explica los tres modos con los ejemplos proyectados:
    1. **Zero-shot:** Solo das la orden en lenguaje natural, sin ejemplos.
    2. **One-shot:** Das la orden más 1 ejemplo resuelto.
    3. **Few-shot:** Incluyes 2 a 5 ejemplos `(entrada → salida esperada)` dentro del mismo prompt para que el modelo deduzca al vuelo tanto la lógica como el formato exacto.
- **Respaldo técnico para el profesor (cifras reales):**
  - En **TriviaQA (Brown et al., 2020)**, GPT-3 subió de **64.3%** (*zero-shot*) a **68.0%** (*one-shot*) y **71.2%** (*few-shot*).
  - **¿Y qué pasa cuando en empresa SÍ hace falta modificar pesos sin gastar una fortuna?** Existe **PEFT / LoRA** (*Low-Rank Adaptation*, **Huyen, 2025, pp. 716–735; Chen & Liu, 2024, p. 33**): un modelo 7B exige ~56 GB de VRAM en *full fine-tuning* (14 GB pesos + 42 GB gradientes y estados de Adam) y sufre *olvido catastrófico*; LoRA congela los pesos originales y entrena pequeñas matrices auxiliares (<1% —en GPT-3 175B apenas 4.7M–18.8M, **0.0027%–0.01%**— sin añadir latencia al fusionarse). Úsalo solo si algún estudiante pregunta *"¿pero nunca se reentrena nada?"*.

## Diapositiva 11 — Demostración Práctica y Analogía del Actor *(58 – 65 min)*
- **Tiempo:** `58 – 65 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Comparativa lado a lado del prompt ambiguo vs. el prompt estructurado (*few-shot* + *"Pensemos paso a paso"*).
  - *Clic 2:* Tarjeta derecha — La Analogía del Actor y el Libreto.
- **Guion y dinámica docente:**
  - **58 – 65 min | Demostración práctica en vivo (o proyectando la diapositiva):**
    - Si tienes acceso en vivo a un chat de IA, o usando el caso proyectado: muestra cómo un **prompt ambiguo y sin ejemplos** (*"Resume este reporte para la dirección"*) produce un texto genérico o mal estructurado.
    - Luego aplica la versión **few-shot + razonamiento guiado (*«Pensemos paso a paso» / Chain-of-Thought*)**: le das 2 ejemplos de formato exacto o le pides analizar datos antes de concluir. La calidad, consistencia y precisión lógica saltan drásticamente (**Wei et al., 2022, p. 5; Huyen, 2025, pp. 498–502** explican que generar pasos intermedios asigna más "cómputo en tiempo de prueba" antes de comprometerse con la respuesta final).
  - **La analogía del actor:** El modelo ya completó años de "escuela de actuación" durante su preentrenamiento. Cuando tú le escribes un buen prompt, **eres el director entregándole el libreto 5 minutos antes de salir a escena**: el actor no regresa a estudiar actuación (no se reentrena), solo lee las indicaciones e interpreta el personaje con precisión.

## Diapositiva 12 — Desmitificación Clave: ¿El Chat Aprende Para Siempre? *(65 – 68 min)*
- **Tiempo:** `65 – 68 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Mito popular ("Cuando chateo con la IA, aprende y guarda mis datos en su cerebro").
  - *Clic 2:* Realidad técnica + diagrama de Memoria de Trabajo (Ventana de Contexto) vs. Cerebro Base Congelado.
- **Guion y dinámica docente:**
  - Aclara el error conceptual más extendido entre usuarios e incluso ejecutivos: **«Chatear con el modelo no lo reentrena ni modifica su cerebro para siempre.»**
  - Explica el diagrama: durante un chat, todas tus preguntas y sus respuestas viven temporalmente en la **memoria de trabajo activa** (la ventana de contexto que se reenvía en cada mensaje). El cerebro base (los pesos de la red) permanece **congelado e idéntico** para todos los usuarios. Si abres un chat nuevo, la memoria de trabajo empieza en blanco otra vez.
  - *(Matiz práctico para el profesor si preguntan por "Memoria" en ChatGPT o privacidad:* Funciones como "Memoria personalizada" o RAG guardan texto en una base de datos externa para inyectarlo otra vez en el futuro prompt, **no** modifican los pesos neuronales del modelo; y el uso de chats para entrenar futuras versiones ocurre solo en procesos *offline* posteriores si las políticas de datos lo permiten, nunca en tiempo real mientras chateas).*

---

# BLOQUE 5: La Realidad Práctica: Riesgos, Alineación y Homogeneización (68 – 82 min)

> **Objetivo:** Aterrizar el pensamiento crítico: comprender las limitaciones, la seguridad y los desafíos reales de ingeniería.

## Diapositiva 13 — Del Modelo Crudo al Asistente: La Máscara de Alineación *(68 – 74 min)*
- **Tiempo:** `68 – 74 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Etapa 1 (Preentrenamiento: el reflejo crudo de todo internet, sin filtros ni modales).
  - *Clic 2:* Etapas 2 y 3 (SFT y Alineación RLHF/DPO = colocar la máscara amable de asistente).
  - *Clic 3:* Advertencia central ("La máscara guía el tono, pero no borra el fondo").
- **Guion y dinámica docente:**
  - Explica qué sale realmente del preentrenamiento masivo (**Huyen, 2025, pp. 188–189; Chen & Liu, 2024, pp. 19–20**): sale un **completador estadístico crudo** que absorbió todo internet sin filtros ni modales —incluyendo conocimientos brillantes, pero también contradicciones, sesgos humanos, toxicidad e inventos plausibles (alucinaciones). Si le preguntas cómo hacer algo, un modelo base puro podría simplemente seguir autocompletando más preguntas de un foro en vez de ayudarte.
  - Por eso existe el **post-entrenamiento** (que consume apenas ~2% del cómputo total frente al 98% del preentrenamiento, **Huyen, p. 186**):
    1. **SFT (*Supervised Fine-Tuning*):** Se le muestran decenas de miles de conversaciones modelo `(pregunta, respuesta ideal)` para enseñarle el oficio de asistente. *(Dato real InstructGPT: ~13,000 demostraciones redactadas por expertos, \$10–\$25 cada una, **Huyen, pp. 193–194**).*
    2. **Alineación con Preferencias Humanas (`RLHF` / `DPO`):** Evaluadores humanos comparan pares de respuestas (¿cuál es más útil, honesta y segura?) y se entrena al modelo para preferir ese estilo. *(Nota docente: RLHF con PPO usa un modelo de recompensa separado basado en Bradley-Terry; DPO —usado en Llama 3— simplifica esto optimizando directamente las probabilidades sin modelo de recompensa aparte, **Chen & Liu, pp. 32–33; Huyen, pp. 196–204**).*
  - **La analogía (Nota docente — sin mencionar nombres extraños en las diapositivas):** En la diapositiva mostramos la metáfora visual de **ponerle una máscara amable a una criatura caótica** *(en la comunidad técnica de IA suele representarse con el meme del "Shoggoth" con una carita feliz, Huyen Fig. 2-11, p. 188 — menciónalo verbalmente solo si conviene, manteniéndolo limpio en pantalla)*. Lo crucial es el mensaje pedagógico: la máscara corrige las formas y bloquea respuestas dañinas obvias, **pero no elimina de raíz** los sesgos profundos ni la tendencia a alucinar que viven dentro del modelo base.

## Diapositiva 14 — Homogeneización y Punto Único de Falla *(74 – 79 min)*
- **Tiempo:** `74 – 79 min`
- **Secuencia de clics en pantalla:**
  - *Clic 1:* Diagrama de cascada — Pocos modelos base alimentando Medicina, Finanzas, Educación y Atención al Cliente (Punto Único de Falla).
  - *Clic 2:* Tarjetas derechas sobre propagación de errores y fragilidad ante manipulaciones (*jailbreaks*).
- **Guion y dinámica docente:**
  - Retoma el concepto de **homogeneización** de **Bommasani et al. (2021, §1.1, pp. 3–6 y §4.7, pp. 105–107)**: hoy casi todo el planeta construye sus aplicaciones críticas encima de los mismos 3 o 4 modelos base.
  - Esto produce enorme eficiencia, **pero convierte al Modelo Fundacional en un Punto Único de Falla (*Single Point of Failure*)**: cualquier sesgo, punto ciego o vulnerabilidad en el cimiento se hereda automáticamente y en cascada hacia millones de usuarios en medicina, finanzas, educación o justicia.
  - Explica la **fragilidad de seguridad:** incluso modelos bien alineados pueden ser engañados mediante técnicas de ingeniería social en texto (*jailbreaks*, cambios de rol/formato o instrucciones maliciosas ocultas en correos/documentos que el asistente lee — *inyección indirecta*). Unos pocos ejemplos manipulados bastan para perforar la máscara amable.
- **Respaldo técnico para el profesor (cifras de impacto):**
  - *Evaluación y Ley de Goodhart (**Bommasani et al., p. 96; Huyen, pp. 420–437**):* Cuando un puntaje público (benchmark como MMLU, GSM8K o HumanEval) se vuelve objetivo de marketing, deja de medir inteligencia real debido a **contaminación de datos** (GPT-3 tuvo >40% de solapamiento en 13 benchmarks vía 13-gramas, **Brown et al., p. 7**) o **sobreajuste** al formato del examen.
  - *Correlación inteligencia vs. veracidad:* En la Tabla 4-5 de Huyen (p. 427), MMLU correlaciona fuertemente con lógica ($r=0.87\text{–}0.90$), pero apenas $r=0.42\text{–}0.50$ con **TruthfulQA** — un modelo más inteligente no necesariamente alucina menos.
  - *Auditoría ofensiva (Red-Teaming) y data poisoning:* En pruebas automatizadas (PAIR, Chao et al., 2023; **Huyen, p. 526**), un agente atacante logra *jailbreaks* en menos de 20 intentos. Y en entrenamiento web sin curación, modificar apenas **2 de cada 3 millones de ejemplos (0.000067%)** basta para implantar una puerta trasera en modelos estilo CLIP (**Carlini & Terzis, 2021, en Bommasani et al., p. 106**).

## Diapositiva 15 — Conclusión Operativa: Ingeniería Confiable *(79 – 82 min)*
- **Tiempo:** `79 – 82 min`
- **Secuencia de clics en pantalla:**
  - *Clics 1, 2 y 3:* Los tres pilares de defensa alrededor del modelo (**RAG / Conocimiento verificado**, **Verificación de fuentes** y **Validación humana**).
- **Guion y dinámica docente:**
  - Aterriza el bloque con un mensaje práctico: en la vida real, nadie serio despliega un Modelo Fundacional "a pelo" confiando ciegamente en su memoria paramétrica.
  - Explica los tres cinturones de seguridad operativos:
    1. **RAG (*Retrieval-Augmented Generation*):** En vez de dejar que el modelo improvise desde su memoria difusa, primero se buscan documentos oficiales/actualizados y se le entregan en el contexto para que responda con base en ellos.
    2. **Verificación de fuentes:** Exigir citas verificables, reglas automáticas de chequeo y tests unitarios (en código) antes de aceptar una salida.
    3. **Validación humana (*Human-in-the-Loop*):** Mantener a un profesional con criterio revisando decisiones sensibles (diagnósticos, contratos, pagos).

---

# BLOQUE 6: Síntesis, Preguntas y Cierre (82 – 90 min)

> **Objetivo:** Consolidar el mapa mental global de la clase y responder dudas finales.

## Diapositiva 16 — Recapitulación: El Mapa Conceptual Completo *(82 – 86 min)*
- **Tiempo:** `82 – 86 min`
- **Secuencia de clics en pantalla:**
  - *Clics 1 al 5:* Se iluminan en secuencia los 5 peldaños del recorrido completo (**Base → Escala → Emergencia → Uso → Cuidado**).
- **Guion y dinámica docente:**
  - Recorre los 5 puntos como una sola historia conectada:
    1. **Base:** Preentrenamos un único modelo general con datos masivos sin etiquetas (autosupervisión), creando nuestro "bloque de mármol".
    2. **Escala:** Descubrimos que debe crecer balanceando parámetros y datos (~20 tokens por parámetro, Chinchilla) y pensando en el costo real de uso en producción.
    3. **Emergencia:** Al cruzar ciertos umbrales cuantitativos, despiertan capacidades cualitativas nuevas que nadie programó explícitamente.
    4. **Uso:** Lo guiamos al instante desde el contexto con instrucciones, ejemplos y razonamiento paso a paso, sin tocar sus pesos internos.
    5. **Cuidado:** Le colocamos capas de alineación sabiendo que la máscara amable no borra los riesgos de fondo, y lo rodeamos de fuentes verificadas y criterio humano.

## Diapositiva 17 — Ronda de Preguntas Abiertas y Cierre *(86 – 90 min)*
- **Tiempo:** `86 – 90 min`
- **Secuencia de clics en pantalla:**
  - Proyecta la pregunta de salida limpia en pantalla mientras moderas el diálogo final.
- **Guion y dinámica docente:**
  - Abre el micrófono para dudas generales sobre cualquier bloque y cierra lanzando la **pregunta de salida** para que el grupo se lleve la reflexión central:
    > *«Ahora que conocen cómo se construye un modelo fundacional, ¿dónde creen que una empresa debería poner más cuidado al adoptarlo: en el tamaño del modelo o en cómo lo guía y valida en el contexto?»*
  - **Cierre sugerido tras escuchar 2–3 respuestas:** Aunque durante años el mercado se obsesionó con el tamaño bruto del modelo, hoy la diferencia real entre un proyecto fallido y uno valioso está en el segundo polo: **qué contexto verificado le das (RAG / buenos ejemplos), cómo validas sus salidas y qué supervisión humana pones donde el riesgo importa.**
