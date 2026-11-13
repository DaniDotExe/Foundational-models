# Guion y Notas del Orador — Foundation Models (Enlazado 1 a 1 con las Diapositivas)

> **Documento de cátedra exclusivo para el profesor.** Cada sección está enlazada 1 a 1 con su diapositiva correspondiente en [`presentacion_foundation_models.html`](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html). Además, dentro de la presentación puedes pulsar la tecla **`P`** para abrir la ventana sincronizada de *Presenter View*, o hacer clic en el contador inferior derecho (`NN / 48`) de cualquier lámina para saltar directamente a su nota en [`notas_del_orador.html`](file:///d:/Software/Foundational%20models/notas_del_orador.html).

## Estrategia Pedagógica, Distribución de los 90 Minutos y Criterio de Curaduría

| Bloque Estructural | Rango de Diapositivas | Tiempo Asignado | Núcleo Fundamental (Profundidad Alta) | Mención Contextual / Omisiones Deliberadas |
| :--- | :--- | :--- | :--- | :--- |
| **I. El Cambio de Paradigma** | [Slides 01 – 09](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-1) | **15 min** (`00:00–15:00`) | Evolución ML → DL → FM (Build 3 pasos), definición CRFM (Bommasani 2021), Emergencia y Homogeneización, SSL (CLM, MLM, InfoNCE, Diffusion). | Se omite historia pre-2005 (sistemas expertos simbólicos) y derivaciones de SVM. |
| **II. Tokenización y Motor Transformer** | [Slides 10 – 21](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-10) | **24 min** (`15:00–39:00`) | BPE, RoPE, construcción paso a paso de Atención $\mathbf{Q},\mathbf{K},\mathbf{V}$ (Build 4 pasos), máscara causal vs. bidireccional, BERT (Devlin 2019), GPT-3 (Brown 2020), muestreo $T$/Top-$p$ y CLIP/ViT. | Se omiten variantes de atención lineal obsoletas (Linformer/Performer) para priorizar FlashAttention y GQA. |
| **III. Leyes de Escala y Emergencia** | [Slides 22 – 31](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-22) | **18 min** (`39:00–57:00`) | Leyes de potencia de Kaplan et al. (2020) (Build 3 pasos), frontera Compute-Optimal de Chinchilla (Hoffmann 2022, $D \approx 20N$), sobre-entrenamiento Llama-3 y Emergencia (Wei 2022 vs. Schaeffer 2023). | Se omiten demostraciones de física estadística de campos medios; se prioriza el lagrangiano $C=6ND$. |
| **IV. Post-Entrenamiento e Ingeniería de IA** | [Slides 32 – 41](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-32) | **17 min** (`57:00–74:00`) | Pipeline SFT → RLHF → DPO (Build 3 pasos), matriz Prompting vs. RAG vs. Fine-Tuning (Huyen 2025), RAG híbrido + Reranker, Agentes ReAct, LoRA/QLoRA e inferencia Prefill/Decode + KV-Cache. | Se omite la derivación completa de TRPO/PPO con ventaja generalizada (GAE) en favor de la equivalencia cerrada de DPO. |
| **V. Evaluación, Seguridad y Gobernanza** | [Slides 42 – 48](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-42) | **16 min** (`74:00–90:00`) | Perplejidad, contaminación de benchmarks, LLM-as-a-Judge, Tríada RAG, alucinaciones y descalibración post-RLHF, Prompt Injection, Muro de Datos y síntesis de 5 ecuaciones. | — |

---

## I. El Cambio de Paradigma: De Task-Specific ML a Foundation Models

### <a id="nota-1"></a>Nota 01 — [Foundation Models: De la Predicción Estadística a la Ingeniería de Sistemas Inteligentes](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-1)
- **Diapositiva enlazada:** [Abrir Diapositiva #01 (`#slide-1`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-1)
- **Tiempo interno estimado:** `Min 00:00 – 02:00 (2 min)`
- **Objetivo pedagógico:** Establecer la tesis central de la cátedra: cómo un objetivo estadístico simple (auto-supervisión a escala) transformó la inteligencia artificial en una disciplina de infraestructura y sistemas.
- **Guion verbal del profesor:**
  > Bienvenidos. Hoy vamos a diseccionar rigurosamente qué es un Modelo Fundacional, por qué funciona desde el punto de vista matemático y computacional, y cómo se transforma un predictor estadístico crudo en un sistema de ingeniería confiable. No los trataremos como cajas negras mágicas: recorreremos cuatro ejes estructurales —el cambio de paradigma hacia la auto-supervisión, la mecánica interna del Transformer, la física empírica de las leyes de escalamiento y la ingeniería post-entrenamiento e inferencia.
- **Pregunta socrática a la audiencia:** *Antes de iniciar: si tuvieran que definir en una sola frase qué diferencia estructuralmente a un Foundation Model de una red neuronal profunda tradicional como ResNet-50, ¿dónde trazarían la frontera?*
- **Puente hacia la Diapositiva 02:** Para responder con precisión a esa pregunta, veamos la estructura de los cinco módulos que recorreremos hoy.

### <a id="nota-2"></a>Nota 02 — [Arquitectura Conceptual de la Sesión](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-2)
- **Diapositiva enlazada:** [Abrir Diapositiva #02 (`#slide-2`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-2)
- **Tiempo interno estimado:** `Min 02:00 – 04:00 (2 min)`
- **Objetivo pedagógico:** Presentar la hoja de ruta lógica en cinco bloques interconectados y las tres tesis rectoras.
- **Guion verbal del profesor:**
  > Nuestra sesión se articula en cinco bloques progresivos. Primero analizaremos por qué la industria abandonó el entrenamiento de modelos aislados desde cero. En el segundo bloque abriremos el motor matemático del Transformer y los objetivos de pre-entrenamiento. En el tercer bloque estudiaremos las leyes empíricas de escala de Kaplan y Chinchilla junto con el fenómeno de emergencia. En el cuarto bloque pasaremos del modelo base al sistema en producción: SFT, RLHF, DPO, RAG, LoRA e inferencia eficiente. Y cerraremos en el quinto bloque con evaluación rigurosa, vulnerabilidades y riesgos sociotécnicos.
- **Pregunta socrática a la audiencia:** *¿En cuál de estos cinco eslabones creen que fracasan hoy la mayoría de los proyectos empresariales de IA: en la capacidad del modelo base o en la arquitectura del sistema alrededor del modelo?*
- **Puente hacia la Diapositiva 03:** Iniciemos una secuencia progresiva de tres láminas (Slides 3, 4 y 5) para visualizar cómo evolucionó la ingeniería del aprendizaje automático.

### <a id="nota-3"></a>Nota 03 — [Evolución de Paradigmas (1/3): Ingeniería Manual de Características](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-3)
- **Diapositiva enlazada:** [Abrir Diapositiva #03 (`#slide-3`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-3)
- **Tiempo interno estimado:** `Min 04:00 – 05:30 (1.5 min)`
- **Objetivo pedagógico:** Explicar el primer peldaño de la secuencia progresiva: el cuello de botella de la ingeniería manual de características en el ML clásico.
- **Guion verbal del profesor:**
  > Observen la primera columna iluminada en nuestro diagrama arquitectónico. Entre 2005 y 2012, construir un sistema de visión o procesamiento de lenguaje natural implicaba que el investigador diseñara a mano los extractores de características: filtros SIFT o HOG en visión, diccionarios morfológicos y TF-IDF en texto. El algoritmo de aprendizaje —un SVM o un Gradient Boosting— sólo ajustaba una frontera de decisión final. Resolver N tareas exigía construir y mantener N pipelines completamente independientes.
- **Pregunta socrática a la audiencia:** *¿Por qué este paradigma no podía escalar aunque multiplicáramos por mil la potencia de cómputo disponible?*
- **Puente hacia la Diapositiva 04:** Al avanzar a la siguiente lámina veremos cómo el Deep Learning clásico rompió el cuello de botella de las features manuales, pero introdujo un nuevo límite.

### <a id="nota-4"></a>Nota 04 — [Evolución de Paradigmas (2/3): Deep Learning Supervisado por Tarea](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-4)
- **Diapositiva enlazada:** [Abrir Diapositiva #04 (`#slide-4`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-4)
- **Tiempo interno estimado:** `Min 05:30 – 07:00 (1.5 min)`
- **Objetivo pedagógico:** Mostrar el segundo paso de la construcción progresiva: el auge del Deep Learning supervisado y el cuello de botella de la anotación humana.
- **Guion verbal del profesor:**
  > Al pasar a la Fase 2 (2013–2017), las redes neuronales profundas reemplazaron la ingeniería manual de características por aprendizaje jerárquico de representaciones extremo a extremo. Sin embargo, el paradigma seguía siendo específico por tarea: un modelo para análisis de sentimiento, otro distinto para traducción, otro para reconocimiento de entidades. El cuello de botella se desplazó del diseño de features al costo económico y temporal de etiquetar millones de ejemplos supervisados para cada nueva aplicación.
- **Pregunta socrática a la audiencia:** *¿Qué sucede cuando queremos aplicar IA a un dominio especializado —como oncología clínica o derecho tributario— donde simplemente no existen millones de pares etiquetados?*
- **Puente hacia la Diapositiva 05:** Avancemos un paso más sobre este mismo esquema para iluminar la tercera columna: la irrupción de los Modelos Fundacionales.

### <a id="nota-5"></a>Nota 05 — [Evolución de Paradigmas (3/3): Centralización y Adaptación Fundacional](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-5)
- **Diapositiva enlazada:** [Abrir Diapositiva #05 (`#slide-5`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-5)
- **Tiempo interno estimado:** `Min 07:00 – 09:00 (2 min)`
- **Objetivo pedagógico:** Completar la secuencia progresiva mostrando la inversión arquitectónica: de N redes independientes a 1 modelo fundacional centralizado adaptable a N tareas.
- **Guion verbal del profesor:**
  > Aquí se completa la transición. En la Fase 3 desacoplamos la adquisición de conocimiento general de la especialización en tareas. Primero entrenamos un único modelo masivo mediante objetivos auto-supervisados (Self-Supervised Learning, SSL) donde la señal de supervisión proviene de la propia estructura del dato no etiquetado. Ese modelo base concentra una representación densa del mundo y luego se adapta con costo marginal a cientos de tareas downstream.
- **Pregunta socrática a la audiencia:** *¿Qué ventaja económica y de ingeniería ofrece amortizar el costo de pre-entrenamiento de un único backbone sobre cientos de productos distintos?*
- **Puente hacia la Diapositiva 06:** Formalicemos ahora la definición canónica acuñada por el grupo CRFM de Stanford en Bommasani et al. (2021).

### <a id="nota-6"></a>Nota 06 — [Definición Canónica y los Dos Pilares de Bommasani et al. (2021)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-6)
- **Diapositiva enlazada:** [Abrir Diapositiva #06 (`#slide-6`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-6)
- **Tiempo interno estimado:** `Min 09:00 – 11:00 (2 min)`
- **Objetivo pedagógico:** Desglosar la definición formal de Bommasani et al. (2021) y sus dos propiedades definitorias: Emergencia y Homogeneización.
- **Guion verbal del profesor:**
  > En agosto de 2021, más de cien investigadores de Stanford liderados por Rishi Bommasani publicaron 'On the Opportunities and Risks of Foundation Models'. Eligieron la palabra 'fundacional' deliberadamente: estos modelos operan como cimientos de ingeniería civil. Su tesis descansa sobre dos pilares: Emergencia —las capacidades de alto nivel surgen implícitamente al escalar el pre-entrenamiento— y Homogeneización —prácticamente toda la comunidad convergió hacia la arquitectura Transformer y hacia unos pocos modelos base. Eso genera un apalancamiento extraordinario, pero también el riesgo de monocultivo algorítmico.
- **Pregunta socrática a la audiencia:** *¿En qué otra industria tecnológica hemos visto que la homogeneización sobre un único núcleo de software cree tanto eficiencia masiva como vulnerabilidades sistémicas?*
- **Puente hacia la Diapositiva 07:** Aclaremos ahora una confusión frecuente incluso en literatura técnica: ¿es lo mismo un Modelo Fundacional que la IA Generativa?

### <a id="nota-7"></a>Nota 07 — [Fronteras Conceptuales: Foundation Models vs. IA Generativa](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-7)
- **Diapositiva enlazada:** [Abrir Diapositiva #07 (`#slide-7`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-7)
- **Tiempo interno estimado:** `Min 11:00 – 13:00 (2 min)`
- **Objetivo pedagógico:** Demostrar interactivamente que 'Foundation Model' y 'Generative AI' son conjuntos secantes pero no idénticos.
- **Guion verbal del profesor:**
  > Interactuemos con este diagrama de Venn en pantalla. En el discurso público suele usarse 'IA Generativa' y 'Modelo Fundacional' como sinónimos, pero técnicamente son ortogonales. Si hacemos clic en la región izquierda ('FM No Generativos'), encontramos modelos como BERT, RoBERTa, CLIP o DINOv2: son modelos fundacionales puros entrenados con auto-supervisión masiva y adaptables a decenas de tareas, pero su objetivo es producir representaciones latentes o clasificación, no generar muestras abiertas. En el extremo derecho tenemos generadores clásicos específicos de dominio como StyleGAN2 entrenado en caras. Y en la intersección central verde residen los sistemas que combinan ambos mundos: GPT-4, Claude, Llama 3, Gemini o Sora.
- **Pregunta socrática a la audiencia:** *Si su objetivo de ingeniería es indexar 50 millones de documentos legales para búsqueda semántica en menos de 10 milisegundos, ¿elegirían un modelo de la región izquierda (Encoder/Bi-Encoder) o un modelo generativo pesado de la intersección?*
- **Puente hacia la Diapositiva 08:** Veamos cómo la llave maestra que habilitó todo el círculo azul de los Modelos Fundacionales es el Aprendizaje Auto-Supervisado (SSL).

### <a id="nota-8"></a>Nota 08 — [El Motor del Pre-entrenamiento: Aprendizaje Auto-Supervisado (SSL)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-8)
- **Diapositiva enlazada:** [Abrir Diapositiva #08 (`#slide-8`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-8)
- **Tiempo interno estimado:** `Min 13:00 – 15:00 (2 min)`
- **Objetivo pedagógico:** Unificar bajo una sola formulación matemática las cuatro familias de objetivos auto-supervisados (CLM, MLM, InfoNCE y Denoising).
- **Guion verbal del profesor:**
  > En esta tabla tenemos las cuatro ecuaciones que gobiernan casi toda la inteligencia artificial contemporánea. ¿Qué tienen en común? En las cuatro, tomamos un objeto crudo x sin etiquetas humanas, ocultamos o transformamos una parte de él y entrenamos a la red parametrizada por theta para predecir o alinear la parte faltante. Ya sea predecir el token futuro dado el pasado (CLM), reconstruir tokens enmascarados con contexto bidireccional (MLM), acercar pares imagen-texto en la hiperesfera unitaria (InfoNCE) o revertir un proceso de ruido gaussiano (Diffusion/Flow Matching), la supervisión es intrínseca al dato.
- **Pregunta socrática a la audiencia:** *¿Por qué creen que una tarea aparentemente tan 'superficial' como adivinar la siguiente palabra termina forzando a la red a aprender razonamiento causal o depuración de código?*
- **Puente hacia la Diapositiva 09:** Antes de entrar al bloque II sobre el Transformer, resumamos el costo de esta centralización: el doble filo de la homogeneización.

### <a id="nota-9"></a>Nota 09 — [El Doble Filo de la Homogeneización Arquitectónica y de Datos](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-9)
- **Diapositiva enlazada:** [Abrir Diapositiva #09 (`#slide-9`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-9)
- **Tiempo interno estimado:** `Min 15:00 – 16:30 (1.5 min)`
- **Objetivo pedagógico:** Cerrar el Bloque I contrastando las ventajas de ingeniería de la homogeneización frente a los riesgos de monocultivo.
- **Guion verbal del profesor:**
  > Cerramos nuestro primer bloque con esta dualidad estructural. En la columna izquierda vemos por qué el mundo adoptó los Modelos Fundacionales: cuando toda la industria converge sobre el bloque Transformer, cualquier avance en compiladores o kernels como FlashAttention beneficia simultáneamente a modelos de lenguaje, visión y proteínas. Pero la columna derecha nos advierte como ingenieros: esa misma convergencia crea un monocultivo. Un defecto estadístico o una vulnerabilidad de seguridad en los cimientos se multiplica por cada aplicación derivada.
- **Pregunta socrática a la audiencia:** *¿Cómo diseñamos entonces una arquitectura empresarial para capturar el apalancamiento de la izquierda mitigando el punto único de fallo de la derecha?*
- **Puente hacia la Diapositiva 10:** Para responderlo necesitamos abrir la caja negra. Entremos al Bloque II: Tokenización, Embeddings y la arquitectura Transformer.

## II. Representación, Tokenización y el Motor Transformer

### <a id="nota-10"></a>Nota 10 — [De Datos Crudos a Vectores: Tokenización Subword (BPE) y Geometría RoPE](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-10)
- **Diapositiva enlazada:** [Abrir Diapositiva #10 (`#slide-10`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-10)
- **Tiempo interno estimado:** `Min 16:30 – 19:00 (2.5 min)`
- **Objetivo pedagógico:** Explicar el pipeline de discretización BPE, la matriz de embeddings y por qué RoPE es indispensable para romper la invarianza a permutaciones del Transformer.
- **Guion verbal del profesor:**
  > Entramos al Bloque II. Una red neuronal sólo opera sobre tensores en R^d. El primer eslabón, muchas veces subestimado, es el tokenizador Byte-Pair Encoding (BPE). Como ilustra el diagrama superior, fusionamos iterativamente los pares de bytes más frecuentes hasta construir un vocabulario de entre 32 mil y 128 mil sub-palabras. Cada ID entero indexa la matriz de embeddings E para obtener un vector denso de dimensión d (por ejemplo, 4096). Pero atención: la operación de atención pura es un operador sobre conjuntos, invariante al orden ('el perro mordió al hombre' produciría lo mismo que 'el hombre mordió al perro'). Por eso los modelos modernos aplican Rotary Position Embeddings (RoPE): rotan los vectores Query y Key un ángulo m*theta para que el producto escalar capture exactamente la distancia relativa m - n.
- **Pregunta socrática a la audiencia:** *¿Por qué un tokenizador entrenado mayoritariamente en inglés penaliza tanto en latencia como en costo financiero a un usuario que interactúa en español, árabe o quechua?*
- **Puente hacia la Diapositiva 11:** Una vez que tenemos nuestra matriz de entrada X de tamaño [longitud l por dimensión d], entremos a una secuencia progresiva de 4 láminas (Slides 11 a 14) para construir paso a paso el bloque Transformer.

### <a id="nota-11"></a>Nota 11 — [Anatomía del Transformer (1/4): Proyecciones Lineales Query, Key y Value](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-11)
- **Diapositiva enlazada:** [Abrir Diapositiva #11 (`#slide-11`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-11)
- **Tiempo interno estimado:** `Min 19:00 – 20:30 (1.5 min)`
- **Objetivo pedagógico:** Introducir el Paso 1 del grafo computacional del Transformer: la descomposición de cada token en los subespacios Query, Key y Value.
- **Guion verbal del profesor:**
  > Iniciamos nuestra segunda secuencia progresiva de cuatro láminas sobre el corazón del Transformer (Vaswani et al., 2017). En este Paso 1 iluminado a la izquierda, la matriz de entrada X de dimensión [l por d] se multiplica por tres matrices de pesos aprendidas: W_Q, W_K y W_V. La analogía exacta proviene de los sistemas de recuperación de información y memorias asociativas diferenciables: Query representa qué busca el token actual; Key representa bajo qué etiqueta se indexa cada token del contexto; y Value contiene el vector de información que se transferirá cuando haya coincidencia entre Query y Key.
- **Pregunta socrática a la audiencia:** *¿Qué perdería geométricamente la red si obligáramos a que Query y Key compartieran exactamente la misma matriz de proyección (W_Q = W_K)?*
- **Puente hacia la Diapositiva 12:** Si W_Q fuera igual a W_K, la similitud sería obligatoriamente simétrica (el token A atendería a B tanto como B a A). Avancemos a la lámina 12 para ver cómo se calcula y escala esa similitud.

### <a id="nota-12"></a>Nota 12 — [Anatomía del Transformer (2/4): Producto Escalar y Factor de Escala 1/√d_k](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-12)
- **Diapositiva enlazada:** [Abrir Diapositiva #12 (`#slide-12`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-12)
- **Tiempo interno estimado:** `Min 20:30 – 22:00 (1.5 min)`
- **Objetivo pedagógico:** Explicar el Paso 2 del grafo: el cálculo de la matriz de afinidad QK^T y la demostración matemática de por qué se divide por la raíz cuadrada de d_k.
- **Guion verbal del profesor:**
  > Al avanzar al Paso 2 en nuestro esquema, computamos todos los productos escalares entre consultas y claves: S = Q por K transpuesta. Esto produce una matriz de l por l. Pero observen el divisor raíz cuadrada de d_k. No es un detalle cosmético: si las componentes de q y k tienen media cero y varianza unitaria, la suma de d_k productos independientes tiene varianza igual a d_k. Cuando d_k vale 64 o 128, los productos escalares alcanzan magnitudes grandes, haciendo que el Softmax posterior colapse en una distribución one-hot con gradientes prácticamente nulos. Dividir por raíz de d_k restablece la varianza exactamente a 1.
- **Pregunta socrática a la audiencia:** *¿Por qué este paso permite entrenar sobre secuencias de miles de tokens en paralelo en una GPU, mientras que una LSTM debía procesar token por token secuencialmente?*
- **Puente hacia la Diapositiva 13:** Pasemos a la lámina 13 para aplicar la máscara causal, normalizar con Softmax y agregar los vectores Value.

### <a id="nota-13"></a>Nota 13 — [Anatomía del Transformer (3/4): Enmascaramiento, Softmax y Mezcla de Valores](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-13)
- **Diapositiva enlazada:** [Abrir Diapositiva #13 (`#slide-13`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-13)
- **Tiempo interno estimado:** `Min 22:00 – 23:30 (1.5 min)`
- **Objetivo pedagógico:** Explicar el Paso 3: cómo la máscara M preserva la causalidad y cómo la multiplicación por V construye representaciones contextualizadas.
- **Guion verbal del profesor:**
  > En el Paso 3 sumamos la matriz de máscara M antes del Softmax. En un modelo autoregresivo como GPT o Llama, queremos entrenar prediciendo simultáneamente todos los tokens de la secuencia en una sola pasada matricial, pero el token en la posición i jamás puede espiar al token en la posición i+1. Al sumar menos infinito en el triángulo superior (j > i), el exponencial dentro del Softmax se vuelve exactamente cero. Finalmente, multiplicamos esa distribución de probabilidad fila por fila contra la matriz V: cada token emerge como una combinación convexa ponderada de los valores relevantes de su contexto.
- **Pregunta socrática a la audiencia:** *¿Cuál es la diferencia fundamental entre los pesos fijos de un filtro convolucional 3x3 y los pesos de atención A_ij que acabamos de calcular?*
- **Puente hacia la Diapositiva 14:** Completamos nuestro bloque Transformer en la lámina 14 añadiendo Multi-Head Attention, el flujo residual, Pre-RMSNorm y la red SwiGLU.

### <a id="nota-14"></a>Nota 14 — [Anatomía del Transformer (4/4): Multi-Head Attention, Residual Stream y SwiGLU](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-14)
- **Diapositiva enlazada:** [Abrir Diapositiva #14 (`#slide-14`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-14)
- **Tiempo interno estimado:** `Min 23:30 – 25:30 (2 min)`
- **Objetivo pedagógico:** Cerrar la secuencia progresiva explicando la división del trabajo dentro del bloque Transformer: Atención (comunicación entre tokens) vs. FFN SwiGLU (cómputo y memoria por token) + Pre-RMSNorm.
- **Guion verbal del profesor:**
  > Con el Paso 4 completamos el bloque Transformer moderno (estilo Llama-3, Qwen o DeepSeek). Primero, en lugar de una sola atención, ejecutamos h cabezas en paralelo y concatenamos sus salidas. Segundo, adoptamos Pre-RMSNorm antes de cada sub-capa y sumamos el resultado al flujo residual ('Residual Stream'), garantizando que los gradientes fluyan limpiamente a través de más de 100 capas. Y tercero, observen la división funcional del trabajo: el bloque de Atención mueve información horizontalmente entre distintos tokens del contexto, mientras que la red Feed-Forward con activación SwiGLU procesa cada token verticalmente y almacena cerca de dos tercios de todos los parámetros del modelo como un banco de memoria fáctica.
- **Pregunta socrática a la audiencia:** *Si quisieran editar o borrar un hecho puntual almacenado en los pesos de un LLM, ¿en qué matrices buscarían primero: en las proyecciones de atención o en las capas MLP/SwiGLU?*
- **Puente hacia la Diapositiva 15:** Experimentemos ahora en vivo en la lámina 15 con el simulador interactivo de máscaras de atención y escalamiento cuadrático O(l^2).

### <a id="nota-15"></a>Nota 15 — [Simulador de Enmascaramiento Causal vs. Bidireccional y Complejidad O(ℓ²)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-15)
- **Diapositiva enlazada:** [Abrir Diapositiva #15 (`#slide-15`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-15)
- **Tiempo interno estimado:** `Min 25:30 – 28:00 (2.5 min)`
- **Objetivo pedagógico:** Demostrar visual y cuantitativamente cómo la matriz de máscara define las tres familias de arquitecturas Transformer y cómo escala el costo cuadrático con la longitud de secuencia.
- **Guion verbal del profesor:**
  > Probemos el simulador en pantalla. A la izquierda tenemos la matriz de atención de 6 por 6 tokens. En modo 'Causal' (GPT/Llama), todas las celdas por encima de la diagonal principal están bloqueadas con menos infinito: cada fila sólo suma probabilidad sobre su pasado. Si conmutamos al botón 'Bidireccional' (BERT), la matriz entera se ilumina: todos los tokens se ven entre sí simultáneamente. Y en 'Prefix-LM' (T5/UL2), el prefijo de entrada es bidireccional mientras la generación es causal. Ahora observen el control deslizante de la derecha: al llevar la longitud de contexto de 8K a 128K o 1 millón de tokens, el número de pares l^2 crece de millones a un billón (10^12) de interacciones por capa.
- **Pregunta socrática a la audiencia:** *Viendo estos números en tiempo real: ¿por qué ofrecer ventanas de 1 millón de tokens en producción sigue siendo un desafío económico aunque el modelo las soporte arquitectónicamente?*
- **Puente hacia la Diapositiva 16:** Esa diferencia entre la máscara bidireccional y la causal nos lleva directamente a comparar las tres grandes familias arquitectónicas en las siguientes láminas.

### <a id="nota-16"></a>Nota 16 — [Taxonomía de Arquitecturas: Encoder-Only, Decoder-Only y Encoder-Decoder](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-16)
- **Diapositiva enlazada:** [Abrir Diapositiva #16 (`#slide-16`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-16)
- **Tiempo interno estimado:** `Min 28:00 – 30:00 (2 min)`
- **Objetivo pedagógico:** Comparar rigurosamente las tres topologías derivadas del Transformer y explicar por qué Decoder-Only dominó el escalamiento generativo mientras Encoder-Only domina la recuperación.
- **Guion verbal del profesor:**
  > Esta matriz resume la evolución arquitectónica desde 2018. Los modelos Encoder-Only como BERT ven el contexto izquierdo y derecho simultáneamente, lo que los convierte en extractores de representaciones insuperables por unidad de cómputo. Los modelos Encoder-Decoder como T5 o Whisper brillan cuando mapeamos una modalidad de entrada a otra secuencia de salida. Sin embargo, para modelos fundacionales generales de gran escala, la arquitectura Decoder-Only ganó la carrera por dos razones matemáticas: aprovecha el 100% de los tokens para calcular gradientes de predicción —no sólo el 15% enmascarado— y reduce cualquier tarea a continuación de texto en contexto.
- **Pregunta socrática a la audiencia:** *Si están diseñando un clasificador de fraude que debe evaluar 20,000 transacciones por segundo con presupuesto de latencia de 5 ms, ¿desplegarían un Decoder de 70B o un Encoder destilado de 300M?*
- **Puente hacia la Diapositiva 17:** Examinemos con lupa los dos artículos fundacionales de estas familias: primero BERT (Devlin et al., 2019) y luego GPT-3 (Brown et al., 2020).

### <a id="nota-17"></a>Nota 17 — [El Paradigma Bidireccional: BERT y Masked Language Modeling (Devlin et al., 2019)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-17)
- **Diapositiva enlazada:** [Abrir Diapositiva #17 (`#slide-17`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-17)
- **Tiempo interno estimado:** `Min 30:00 – 31:30 (1.5 min)`
- **Objetivo pedagógico:** Explicar la mecánica exacta del Masked Language Modeling en BERT, la regla 80/10/10 y su supuesto de independencia condicional.
- **Guion verbal del profesor:**
  > En nuestro documento de referencia de Devlin et al. (2019), BERT demostró que para tareas de comprensión profunda del lenguaje es crucial fusionar el contexto izquierdo y derecho en todas las capas. Para evitar que un token se copie a sí mismo a través de conexiones multicapa, BERT enmascara el 15% de los tokens e intenta reconstruirlos. Observen la elegancia de ingeniería de la regla 80/10/10: si siempre usáramos el símbolo [MASK], el modelo sufriría un desfase ('pretrain-finetune discrepancy') porque en producción jamás verá el símbolo [MASK]. Mantener un 10% aleatorio y un 10% intacto obliga al Encoder a construir una representación contextual densa para cada palabra de la oración.
- **Pregunta socrática a la audiencia:** *¿Por qué la asunción de que los tokens enmascarados se predicen de forma independiente entre sí impide que BERT redacte ensayos largos con coherencia autoregresiva?*
- **Puente hacia la Diapositiva 18:** Veamos cómo la familia GPT resolvió la generación universal mediante la factorización autoregresiva exacta.

### <a id="nota-18"></a>Nota 18 — [El Paradigma Autoregresivo: Factorización de Cadena y GPT-3 (Brown et al., 2020)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-18)
- **Diapositiva enlazada:** [Abrir Diapositiva #18 (`#slide-18`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-18)
- **Tiempo interno estimado:** `Min 31:30 – 33:30 (2 min)`
- **Objetivo pedagógico:** Explicar la factorización exacta de la regla de la cadena en los modelos GPT y el descubrimiento del In-Context Learning en Brown et al. (2020).
- **Guion verbal del profesor:**
  > A diferencia del modelo enmascarado de BERT, la ecuación de los modelos Decoder-Only como GPT-3 descansa en una identidad matemática exacta: la regla de la cadena de la probabilidad. La probabilidad de cualquier secuencia de texto, código o diálogo es exactamente el producto de las probabilidades condicionales de cada token dado todo su pasado. Cuando Brown et al. (2020) escalaron este objetivo de 1.5 mil millones de parámetros (GPT-2) a 175 mil millones (GPT-3), descubrieron que ya no hacía falta actualizar pesos con gradientes para resolver una tarea nueva: bastaba con escribir la descripción o unos pocos ejemplos dentro del prompt para que el modelo adaptara su comportamiento en el propio forward pass.
- **Pregunta socrática a la audiencia:** *¿Cómo es posible que un modelo aprenda una tarea nueva en milisegundos dentro de la ventana de contexto si sus matrices de pesos W están completamente congeladas?*
- **Puente hacia la Diapositiva 19:** Antes de pasar a las leyes de escala que explican ese salto, veamos el detalle matemático de cómo convierte el modelo sus logits finales en texto: el muestreo estocástico.

### <a id="nota-19"></a>Nota 19 — [Mecánica de Decodificación: Del Vector de Logits a la Secuencia Generada](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-19)
- **Diapositiva enlazada:** [Abrir Diapositiva #19 (`#slide-19`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-19)
- **Tiempo interno estimado:** `Min 33:30 – 35:00 (1.5 min)`
- **Objetivo pedagógico:** Explicar por qué la maximización pura (Greedy) degrada la generación de texto y cómo interactúan Temperatura, Top-K y Top-p.
- **Guion verbal del profesor:**
  > Una vez que la última capa del Transformer proyecta el estado oculto contra la matriz del vocabulario (el 'LM Head'), obtenemos un vector de logits z de tamaño |V| (por ejemplo, 128,000 números reales). ¿Cómo elegimos la siguiente palabra? Holtzman et al. (2020) demostraron en 'The Curious Case of Neural Text Degeneration' que elegir siempre el token más probable (Greedy decoding) conduce paradójicamente a bucles repetitivos antinaturales. Por eso en ingeniería ajustamos dos perillas complementarias: la Temperatura T, que reescala los logits antes del Softmax, y el muestreo de núcleo Top-p, que poda dinámicamente la cola larga de tokens improbables.
- **Pregunta socrática a la audiencia:** *Para un agente que genera consultas SQL o JSON estricto contra una base de datos bancaria, ¿qué combinación de Temperatura y Top-p configurarían en producción?*
- **Puente hacia la Diapositiva 20:** Probemos exactamente esa intuición en nuestro tercer módulo interactivo en la lámina 20.

### <a id="nota-20"></a>Nota 20 — [Laboratorio Interactivo: Temperatura (T) y Poda de Núcleo (Top-p)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-20)
- **Diapositiva enlazada:** [Abrir Diapositiva #20 (`#slide-20`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-20)
- **Tiempo interno estimado:** `Min 35:00 – 37:00 (2 min)`
- **Objetivo pedagógico:** Visualizar en tiempo real cómo la temperatura modifica la entropía de Shannon de la distribución del siguiente token y cómo Top-p poda la cola de alucinación.
- **Guion verbal del profesor:**
  > Movamos los controles deslizantes de este laboratorio. Con Temperatura T = 0.2, observen cómo el token principal ('arquitectura') absorbe casi el 95% de la masa de probabilidad y la entropía cae cerca de 0.3 bits: el modelo es casi determinista. Ahora subamos la Temperatura a T = 1.5 con Top-p = 1.0: la distribución se aplana y tokens raros o incoherentes de la cola inferior ganan probabilidad no nula de ser muestreados, disparando las alucinaciones. Pero si mantenemos T = 0.9 y ajustamos Top-p = 0.85, el modelo conserva diversidad creativa entre las tres primeras opciones válidas mientras poda a cero absoluto la cola ruidosa.
- **Pregunta socrática a la audiencia:** *¿Por qué en las APIs de modelos fundacionales se recomienda ajustar Temperatura o Top-p, pero rara vez llevar ambos simultáneamente a valores extremos?*
- **Puente hacia la Diapositiva 21:** Cerramos el Bloque II ampliando el horizonte: ¿cómo sale el Transformer del texto puro hacia la visión, el audio y la biología?

### <a id="nota-21"></a>Nota 21 — [Expansión Multimodal: Parches Visuales (ViT), CLIP y Modelos Visión-Lenguaje](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-21)
- **Diapositiva enlazada:** [Abrir Diapositiva #21 (`#slide-21`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-21)
- **Tiempo interno estimado:** `Min 37:00 – 39:00 (2 min)`
- **Objetivo pedagógico:** Explicar cómo la tokenización de parches (ViT) y el pre-entrenamiento contrastivo (CLIP) permitieron unificar visión y lenguaje en un solo espacio latente.
- **Guion verbal del profesor:**
  > En esta lámina vemos la prueba definitiva de la tesis de homogeneización de Bommasani y Chen et al. (2024). En 2021, Dosovitskiy et al. demostraron con el Vision Transformer (ViT) que si cortamos una imagen en parches de 16x16 píxeles y los tratamos como tokens, el Transformer supera a las redes convolucionales a gran escala. Simultáneamente, Radford et al. presentaron CLIP: dos encoders (uno visual y uno textual) entrenados sobre 400 millones de pares de internet para maximizar el producto escalar de los pares verdaderos en la diagonal de una matriz N por N. Finalmente, los VLMs modernos conectan ese codificador visual mediante una proyección lineal o MLP directamente a la secuencia de entrada del LLM.
- **Pregunta socrática a la audiencia:** *¿Por qué CLIP pudo clasificar imágenes en categorías que jamás había visto en un dataset supervisado (Zero-Shot Image Classification) simplemente comparando la imagen con el prompt 'una foto de un {objeto}'?*
- **Puente hacia la Diapositiva 22:** Hemos comprendido la arquitectura. Pero una arquitectura por sí sola es sólo un recipiente vacío. Entremos al Bloque III: la física empírica de las Leyes de Escalamiento y la Emergencia.

## III. Leyes de Escalamiento (Scaling Laws) y Capacidades Emergentes

### <a id="nota-22"></a>Nota 22 — [La Hipótesis de Escala y la Física Empírica del Aprendizaje Profundo](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-22)
- **Diapositiva enlazada:** [Abrir Diapositiva #22 (`#slide-22`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-22)
- **Tiempo interno estimado:** `Min 39:00 – 41:00 (2 min)`
- **Objetivo pedagógico:** Introducir el Bloque III explicando las tres variables fundamentales (N, D, C) y de dónde proviene la aproximación canónica C = 6ND.
- **Guion verbal del profesor:**
  > Entramos al Bloque III, el corazón cuantitativo de nuestra clase. Antes de 2020, nadie sabía si duplicar el tamaño de una red seguiría reduciendo el error o si se estancaría rápidamente. Kaplan et al. (2020) demostraron que el rendimiento sigue leyes de potencia análogas a las leyes de la termodinámica a lo largo de más de siete órdenes de magnitud. Todo el análisis gira en torno a tres variables: N (número de parámetros), D (número de tokens de entrenamiento) y C (presupuesto computacional en FLOPs). Noten la identidad fundamental C ≈ 6*N*D: en una multiplicación matriz-vector invertimos 2 operaciones en punto flotante (una multiplicación y una suma) por parámetro y por token en el forward pass, y el doble (4 FLOPs) en el backward pass para calcular los gradientes respecto a activaciones y pesos.
- **Pregunta socrática a la audiencia:** *¿Por qué contar con una ley predictiva que relaciona FLOPs con la pérdida final permite a un laboratorio decidir una inversión de 100 millones de dólares haciendo primero experimentos de 10,000 dólares?*
- **Puente hacia la Diapositiva 23:** Construyamos ahora en una secuencia progresiva de tres láminas (Slides 23, 24 y 25) las tres curvas log-log descubiertas por Kaplan et al. (2020).

### <a id="nota-23"></a>Nota 23 — [Leyes de Escalamiento (1/3): Dependencia con el Número de Parámetros N](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-23)
- **Diapositiva enlazada:** [Abrir Diapositiva #23 (`#slide-23`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-23)
- **Tiempo interno estimado:** `Min 41:00 – 42:15 (1.25 min)`
- **Objetivo pedagógico:** Explicar la primera curva de la secuencia progresiva: la ley de potencia respecto al número de parámetros N y la insensibilidad relativa a la forma exacta (profundidad vs. anchura).
- **Guion verbal del profesor:**
  > En el primer panel iluminado a la izquierda graficamos la pérdida de validación L en el eje vertical contra el número de parámetros N en el eje horizontal, ambos en escala logarítmica. Una ley de potencia L(N) = (N_c / N)^alpha_N se transforma en una línea recta perfecta en coordenadas log-log: log(L) = constante - 0.076 * log(N). Además, Kaplan et al. encontraron un resultado sorprendente para los arquitectos de redes: dentro de un rango razonable, la pérdida depende casi exclusivamente del número total de parámetros N, siendo notablemente insensible a la relación exacta entre número de capas (profundidad) y dimensión oculta (anchura).
- **Pregunta socrática a la audiencia:** *Si la pendiente es -0.076 en escala logarítmica, ¿qué ocurre si intentamos escalar únicamente N manteniendo fijo un dataset pequeño D?*
- **Puente hacia la Diapositiva 24:** Si D es fijo y pequeño, la recta se curva y se aplana por sobreajuste. Avancemos a la lámina 24 para iluminar la segunda curva: el escalamiento con los datos D.

### <a id="nota-24"></a>Nota 24 — [Leyes de Escalamiento (2/3): Dependencia con el Volumen de Datos D](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-24)
- **Diapositiva enlazada:** [Abrir Diapositiva #24 (`#slide-24`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-24)
- **Tiempo interno estimado:** `Min 42:15 – 43:30 (1.25 min)`
- **Objetivo pedagógico:** Mostrar el segundo paso de la construcción progresiva: la ley de potencia respecto a los tokens D y la ecuación combinada L(N, D).
- **Guion verbal del profesor:**
  > Al iluminar el segundo panel central, vemos qué ocurre cuando entrenamos un modelo suficientemente grande variando la cantidad de tokens únicos D: nuevamente emerge una línea recta en escala log-log, con exponente alpha_D ≈ 0.095. La ecuación combinada L(N, D) que aparece en la tarjeta central modela la interacción exacta entre ambos cuellos de botella: si aumentamos N indefinidamente pero dejamos D fijo, el término D_c / D domina la suma y el rendimiento deja de mejorar.
- **Pregunta socrática a la audiencia:** *En la práctica real de un laboratorio, ni N ni D son gratuitos: lo que está fijo es el presupuesto de cómputo C en GPUs. ¿Cómo se comporta la pérdida frente a C?*
- **Puente hacia la Diapositiva 25:** Pasemos a la lámina 25 para iluminar el tercer panel: la envolvente de Pareto del cómputo C y la eficiencia muestral.

### <a id="nota-25"></a>Nota 25 — [Leyes de Escalamiento (3/3): Frontera de Cómputo C y Eficiencia Muestral](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-25)
- **Diapositiva enlazada:** [Abrir Diapositiva #25 (`#slide-25`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-25)
- **Tiempo interno estimado:** `Min 43:30 – 45:00 (1.5 min)`
- **Objetivo pedagógico:** Completar la secuencia progresiva de Kaplan et al. (2020) mostrando la envolvente de Pareto L(C) y por qué Kaplan concluyó que debía priorizarse N sobre D.
- **Guion verbal del profesor:**
  > En el tercer panel completamos la trilogía de Kaplan. Cada curva gris individual representa el entrenamiento de un modelo de tamaño distinto; la línea verde punteada que conecta los puntos tangentes inferiores es la frontera eficiente de Pareto: L(C) proporcional a C elevado a -0.050. Además, Kaplan descubrió que los modelos más grandes son más eficientes en muestras (necesitan menos pasos para alcanzar un nivel de pérdida dado). Eso llevó al equipo de OpenAI en 2020 a una conclusión audaz: si duplicas tu presupuesto de cómputo C, invierte casi todo en hacer el modelo más grande (N proporcional a C^0.73) y aumenta poco los datos (D proporcional a C^0.27). Así se diseñó GPT-3 con 175B parámetros entrenado en sólo 300B tokens.
- **Pregunta socrática a la audiencia:** *¿Estaba en lo cierto Kaplan al recomendar escalar N mucho más rápido que D, o había un sesgo experimental oculto en su protocolo de tasa de aprendizaje?*
- **Puente hacia la Diapositiva 26:** Dos años después, Hoffmann et al. (2022) en DeepMind descubrieron el sesgo exacto de Kaplan y reescribieron las reglas con las Leyes de Chinchilla.

### <a id="nota-26"></a>Nota 26 — [La Revolución Chinchilla: Frontera Óptima en Cómputo (Hoffmann et al., 2022)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-26)
- **Diapositiva enlazada:** [Abrir Diapositiva #26 (`#slide-26`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-26)
- **Tiempo interno estimado:** `Min 45:00 – 47:00 (2 min)`
- **Objetivo pedagógico:** Derivar la ecuación paramétrica de Chinchilla L(N,D) = E + A/N^alpha + B/D^beta y demostrar por qué N y D deben escalar equitativamente.
- **Guion verbal del profesor:**
  > Aquí está la corrección histórica de Hoffmann et al. (2022). ¿Por qué Kaplan se equivocó en la proporción entre N y D? Porque en sus experimentos mantuvo fijo un calendario de tasa de aprendizaje largo para todas las corridas, perjudicando artificialmente a los modelos entrenados con más tokens. Cuando DeepMind corrigió el calendario coseno para cada horizonte D y ajustó la función de pérdida aditiva L(N,D) = E + A/N^alpha + B/D^beta (donde E es la entropía irreducible del texto natural), los exponentes alpha (0.34) y beta (0.28) resultaron casi idénticos. Al resolver el multiplicador de Lagrange sujeto a 6ND = C, tanto N óptimo como D óptimo crecen con la raíz cuadrada de C. Resultado práctico: Chinchilla, con 70B parámetros y 1.4 billones de tokens, derrotó a Gopher de 280B usando los mismos FLOPs.
- **Pregunta socrática a la audiencia:** *¿Qué significa físicamente el término constante E ≈ 1.69 nats en la ecuación de Chinchilla y por qué jamás puede llegar a cero?*
- **Puente hacia la Diapositiva 27:** Experimentemos ahora con el simulador interactivo en la lámina 27 para calcular la asignación óptima de N y D bajo cualquier presupuesto de FLOPs.

### <a id="nota-27"></a>Nota 27 — [Simulador Compute-Optimal: Kaplan (2020) vs. Chinchilla (2022)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-27)
- **Diapositiva enlazada:** [Abrir Diapositiva #27 (`#slide-27`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-27)
- **Tiempo interno estimado:** `Min 47:00 – 49:00 (2 min)`
- **Objetivo pedagógico:** Explorar cuantitativamente la brecha entre la asignación de Kaplan y la de Chinchilla a medida que escalamos de 10^21 a 10^25 FLOPs.
- **Guion verbal del profesor:**
  > Movamos el deslizador de presupuesto computacional C. En el punto de 5.76 x 10^23 FLOPs (el presupuesto de GPT-3), la receta de Kaplan nos pedía construir un monstruo de 175 mil millones de parámetros alimentado con apenas 300 mil millones de tokens (1.7 tokens por parámetro). Al lado derecho, la receta de Chinchilla asigna ese mismo cómputo a un modelo 2.5 veces más compacto (69B parámetros) pero entrenado sobre 1.38 billones de tokens (20 tokens por parámetro). Si subimos el deslizador hasta 10^25 FLOPs (escala de frontera actual), verán cómo un modelo compute-optimal puro requeriría más de 18 billones de tokens de alta calidad, chocando directamente con el 'Data Wall' de texto humano público.
- **Pregunta socrática a la audiencia:** *Si Chinchilla dice que el óptimo es 20 tokens por parámetro, ¿por qué Meta entrenó Llama-3 8B con 15 billones de tokens (casi 1,900 tokens por parámetro, 100 veces más allá de Chinchilla)?*
- **Puente hacia la Diapositiva 28:** Esa pregunta nos lleva directamente a la lámina 28: la era Post-Chinchilla y la amortización del costo de inferencia.

### <a id="nota-28"></a>Nota 28 — [Más Allá de Chinchilla: Sobre-entrenamiento para Inferencia Óptima](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-28)
- **Diapositiva enlazada:** [Abrir Diapositiva #28 (`#slide-28`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-28)
- **Tiempo interno estimado:** `Min 49:00 – 51:00 (2 min)`
- **Objetivo pedagógico:** Explicar por qué la industria moderna sobre-entrena deliberadamente modelos de 7B–70B mucho más allá del ratio 20:1 de Chinchilla para minimizar el costo de inferencia.
- **Guion verbal del profesor:**
  > Esta tabla resuelve la paradoja de por qué ningún modelo abierto moderno respeta el ratio 20 a 1 de Chinchilla. Chinchilla respondió la pregunta: 'Dado un presupuesto fijo para entrenar una sola vez, ¿qué combinación (N, D) minimiza la pérdida?'. Pero en una empresa o servicio en la nube, el entrenamiento se paga una vez y la inferencia se paga miles de millones de veces. Como el costo en FLOPs y memoria de cada token generado en inferencia es proporcional a 2*N e independiente de cuántos tokens D vio el modelo durante su entrenamiento, resulta muchísimo más rentable gastar más cómputo en pre-entrenar un modelo de 8B con 15 billones de tokens (ratio 1,900 a 1) para que sea barato y veloz en producción.
- **Pregunta socrática a la audiencia:** *¿Hasta qué punto sigue bajando la pérdida cuando sobre-entrenamos un modelo pequeño con 100 veces más tokens que Chinchilla, y cuándo empieza a saturarse su capacidad?*
- **Puente hacia la Diapositiva 29:** Mientras la pérdida global baja suavemente como un reloj, en las tareas concretas ocurre algo mucho más dramático: las Capacidades Emergentes (Wei et al., 2022).

### <a id="nota-29"></a>Nota 29 — [Capacidades Emergentes en Grandes Modelos de Lenguaje (Wei et al., 2022)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-29)
- **Diapositiva enlazada:** [Abrir Diapositiva #29 (`#slide-29`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-29)
- **Tiempo interno estimado:** `Min 51:00 – 53:00 (2 min)`
- **Objetivo pedagógico:** Presentar el hallazgo de Wei et al. (2022) sobre transiciones de fase abruptas en tareas downstream al cruzar umbrales críticos de FLOPs.
- **Guion verbal del profesor:**
  > Aquí surge una tensión fascinante en nuestra narrativa. Kaplan y Chinchilla nos acaban de mostrar que la pérdida promedio de entropía cruzada es una recta suave y predecible. Sin embargo, cuando Jason Wei y colegas (2022) midieron la exactitud en tareas específicas como aritmética de tres dígitos, razonamiento multitarea en MMLU o decodificación simbólica en BIG-Bench, encontraron curvas planas pegadas al azar durante cuatro órdenes de magnitud que repentinamente saltan hacia arriba alrededor de 10^22 FLOPs. Igual que el agua líquida cambia abruptamente de fase a hielo a los 0 °C aunque la temperatura baje de forma continua, ciertas capacidades funcionales sólo se manifiestan al superar una escala crítica.
- **Pregunta socrática a la audiencia:** *Si tanto las capacidades útiles (como razonar paso a paso) como los comportamientos riesgosos (como el engaño estratégico) pueden emerger abruptamente al escalar, ¿qué desafío plantea esto para la auditoría de seguridad previa al lanzamiento?*
- **Puente hacia la Diapositiva 30:** Pero, ¿son estas capacidades emergentes una transición de fase física real dentro de los pesos o un artefacto matemático de cómo medimos el éxito? Analicémoslo con nuestro quinto simulador interactivo.

### <a id="nota-30"></a>Nota 30 — [Razonamiento Emergente: Chain-of-Thought y Escalamiento en Tiempo de Inferencia](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-30)
- **Diapositiva enlazada:** [Abrir Diapositiva #30 (`#slide-30`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-30)
- **Tiempo interno estimado:** `Min 53:00 – 54:30 (1.5 min)`
- **Objetivo pedagógico:** Explicar desde la teoría de la complejidad computacional por qué Chain-of-Thought y el Test-Time Compute amplían la clase de problemas resolubles por un Transformer.
- **Guion verbal del profesor:**
  > Detengámonos un minuto en por qué funciona realmente Chain-of-Thought (CoT). No es psicología: es teoría de circuitos computacionales. Cuando le pedimos a un Transformer de 80 capas que responda directamente '¿Cuánto es 748 por 932?' en el primer token, sólo tiene una pasada vertical de 80 capas para hacer toda la aritmética. Al generar tokens intermedios z_1, z_2, ..., z_K ('primero multiplico 748 por 2...'), el modelo utiliza su propia ventana de contexto y el KV-Cache como una cinta de máquina de Turing o memoria RAM externa, invirtiendo O(K * L) operaciones secuenciales antes de emitir la respuesta final.
- **Pregunta socrática a la audiencia:** *¿En qué tipo de aplicaciones empresariales vale la pena pagar 10 segundos y 2,000 tokens de razonamiento interno frente a responder en 200 milisegundos?*
- **Puente hacia la Diapositiva 31:** Resolvamos ahora el debate científico entre Wei et al. (2022) y Schaeffer et al. (2023) en el simulador interactivo de la lámina 31.

### <a id="nota-31"></a>Nota 31 — [Simulador de Emergencia: ¿Transición de Fase o No-Linealidad de la Métrica?](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-31)
- **Diapositiva enlazada:** [Abrir Diapositiva #31 (`#slide-31`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-31)
- **Tiempo interno estimado:** `Min 54:30 – 56:30 (2 min)`
- **Objetivo pedagógico:** Reconciliar la tesis de emergencia (Wei 2022) con la tesis de métricas discontinuas (Schaeffer 2023) mediante simulación matemática de p^K.
- **Guion verbal del profesor:**
  > En NeurIPS 2023, Schaeffer, Miranda y Koyejo publicaron el artículo premiado 'Are Emergent Abilities of Large Language Models a Mirage?'. Su argumento matemático se visualiza en este simulador. Supongamos que la probabilidad p de acertar un solo paso o dígito sube suavemente con la escala: de 0.50 a 0.65, 0.80, 0.90 y 0.96. Si ponemos el deslizador en K = 1 paso, la exactitud sube de forma lineal y predecible. Pero ahora movamos el deslizador a una tarea que exige acertar K = 10 pasos seguidos (Exact Match = p^10): con p = 0.65, 0.65^10 es apenas 1.3% (parece que el modelo no sabe nada); en cambio al llegar a p = 0.95, 0.95^10 salta a 60%. ¿Quién tiene razón? Ambos: el aprendizaje subyacente en entropía cruzada es continuo, pero en el mundo real un programa de código o una cadena lógica de 10 pasos sólo funciona cuando los 10 pasos son correctos.
- **Pregunta socrática a la audiencia:** *Si están evaluando el progreso de un modelo propio durante el entrenamiento y la métrica Exact Match sigue en 0%, ¿qué métrica subyacente deberían monitorear para saber si el entrenamiento va por buen camino?*
- **Puente hacia la Diapositiva 32:** Hemos completado los tres bloques sobre el modelo base. Pero un modelo pre-entrenado crudo no es un asistente ni un producto: entremos al Bloque IV sobre Alineación e Ingeniería de IA.

## IV. Post-Entrenamiento, Alineación e Ingeniería de IA (AI Engineering)

### <a id="nota-32"></a>Nota 32 — [Del Predictor Autoregresivo al Sistema en Producción (Huyen, 2025)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-32)
- **Diapositiva enlazada:** [Abrir Diapositiva #32 (`#slide-32`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-32)
- **Tiempo interno estimado:** `Min 56:30 – 58:00 (1.5 min)`
- **Objetivo pedagógico:** Introducir el marco de Chip Huyen (2025): la brecha entre un modelo base pre-entrenado y un sistema de IA alineado en producción.
- **Guion verbal del profesor:**
  > Entramos al Bloque IV, basado en 'AI Engineering' de Chip Huyen (2025) y los trabajos seminales de alineación (Ouyang et al., 2022; Rafailov et al., 2023). El primer principio de este bloque es entender que un modelo base recién salido del pre-entrenamiento no es un asistente: es un simulador estadístico de documentos de internet. Si le preguntas algo, es igual de probable que te responda con cinco preguntas más porque cree que está autocompletando un examen universitario. Transformarlo en un sistema confiable requiere dos capas: el pipeline de post-entrenamiento (que veremos en una secuencia progresiva en las láminas 33, 34 y 35) y la arquitectura de ingeniería de contexto e inferencia.
- **Pregunta socrática a la audiencia:** *¿Por qué el post-entrenamiento consume menos del 1% del cómputo del pre-entrenamiento pero determina el 90% de la utilidad percibida por el usuario?*
- **Puente hacia la Diapositiva 33:** Recorramos nuestra cuarta secuencia progresiva (Slides 33 → 34 → 35) sobre las tres etapas de alineación post-entrenamiento.

### <a id="nota-33"></a>Nota 33 — [Alineación Post-Entrenamiento (1/3): Supervised Fine-Tuning (SFT)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-33)
- **Diapositiva enlazada:** [Abrir Diapositiva #33 (`#slide-33`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-33)
- **Tiempo interno estimado:** `Min 58:00 – 59:30 (1.5 min)`
- **Objetivo pedagógico:** Explicar la primera etapa del pipeline de alineación: el ajuste fino supervisado (SFT) sobre demostraciones de instrucciones.
- **Guion verbal del profesor:**
  > En la Etapa 1 iluminada a la izquierda tomamos el modelo base pi_0 y lo entrenamos mediante pérdida autoregresiva estándar, pero calculando el gradiente únicamente sobre los tokens de la respuesta ideal y* dada una instrucción x. Como demostró el paper LIMA ('Less Is More for Alignment', Zhou et al., 2023), la calidad y diversidad de estos ejemplos importa muchísimo más que su cantidad: unas decenas de miles de demostraciones rigurosas bastan para que el modelo cambie su modo operativo de 'autocompletar documentos web' a 'seguir instrucciones estructuradas'.
- **Pregunta socrática a la audiencia:** *¿Por qué el ajuste supervisado por imitación (SFT) por sí solo no basta para alinear completamente un modelo en tareas abiertas o creativas?*
- **Puente hacia la Diapositiva 34:** Porque imitar tokens positivos no le enseña al modelo a distinguir sutilmente qué errores debe evitar cuando genera sus propias muestras. Pasemos a la Etapa 2 en la lámina 34: RLHF.

### <a id="nota-34"></a>Nota 34 — [Alineación Post-Entrenamiento (2/3): RLHF, Modelo Bradley-Terry y Penalización KL](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-34)
- **Diapositiva enlazada:** [Abrir Diapositiva #34 (`#slide-34`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-34)
- **Tiempo interno estimado:** `Min 59:30 – 61:00 (1.5 min)`
- **Objetivo pedagógico:** Explicar la formulación matemática de RLHF: el modelo de preferencias pareadas de Bradley-Terry y el rol crítico de la divergencia KL.
- **Guion verbal del profesor:**
  > Al avanzar a la Etapa 2 en el centro del esquema, aprovechamos una asimetría cognitiva humana: para un experto es mucho más fácil y consistente evaluar cuál de dos respuestas generadas (y_w vs y_l) es mejor que redactar desde cero una respuesta perfecta. Con esos pares entrenamos un modelo de recompensa r_phi minimizando la pérdida de Bradley-Terry. Luego usamos aprendizaje por refuerzo (PPO) para que el generador maximice esa recompensa. Pero observen el término de resta beta por la divergencia de Kullback-Leibler respecto a pi_ref: si quitamos ese freno, el generador descubre rápidamente cómo engañar al modelo de recompensa emitiendo secuencias degeneradas que puntúan alto artificialmente ('Reward Hacking' o Ley de Goodhart).
- **Pregunta socrática a la audiencia:** *¿Cuál es el principal obstáculo de infraestructura e ingeniería al ejecutar PPO con cuatro modelos cargados simultáneamente en memoria GPU (Actor, Crítico, Referencia y Reward Model)?*
- **Puente hacia la Diapositiva 35:** Ese altísimo costo de memoria e inestabilidad de hiperparámetros motivó la Etapa 3 en la lámina 35: Direct Preference Optimization (DPO).

### <a id="nota-35"></a>Nota 35 — [Alineación Post-Entrenamiento (3/3): DPO y Refuerzo con Recompensas Verificables](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-35)
- **Diapositiva enlazada:** [Abrir Diapositiva #35 (`#slide-35`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-35)
- **Tiempo interno estimado:** `Min 61:00 – 62:30 (1.5 min)`
- **Objetivo pedagógico:** Completar la secuencia progresiva explicando la derivación cerrada de DPO (Rafailov et al., 2023) y el auge de RLVR/GRPO en modelos razonadores.
- **Guion verbal del profesor:**
  > En la Etapa 3 completamos el pipeline moderno. Rafailov et al. (2023) en Stanford observaron que el problema de optimización de RLHF con restricción KL tiene una solución analítica cerrada que relaciona biunívocamente la recompensa óptima r(x,y) con el logaritmo del cociente de probabilidades entre la política actual pi_theta y la política de referencia pi_ref. Al sustituir esa identidad directamente en la ecuación de Bradley-Terry, la constante de partición Z(x) se cancela y obtenemos la pérdida DPO: optimizamos preferencias humanas directamente sobre pares (y_w, y_l) sin entrenar jamás un Reward Model separado ni correr PPO. Y cuando la tarea tiene verificación objetiva —como compilar código o verificar una prueba matemática— algoritmos como GRPO usan el verificador determinista como señal de refuerzo directa.
- **Pregunta socrática a la audiencia:** *Cuando ustedes como ingenieros toman un modelo ya alineado con SFT+DPO para construir una aplicación empresarial, ¿qué palanca de adaptación deben usar primero: Prompting, RAG o Fine-Tuning?*
- **Puente hacia la Diapositiva 36:** Respondamos esa pregunta con la matriz de decisión de arquitectura de Chip Huyen (2025) en la lámina 36.

### <a id="nota-36"></a>Nota 36 — [Matriz de Decisión de Ingeniería: Prompting vs. RAG vs. Fine-Tuning](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-36)
- **Diapositiva enlazada:** [Abrir Diapositiva #36 (`#slide-36`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-36)
- **Tiempo interno estimado:** `Min 62:30 – 64:30 (2 min)`
- **Objetivo pedagógico:** Establecer el criterio riguroso de ingeniería para elegir entre Prompting, RAG y Fine-Tuning según el problema sea de conocimiento fáctico o de comportamiento/forma.
- **Guion verbal del profesor:**
  > En consultoría e ingeniería de sistemas de IA, el error más costoso es hacer Fine-Tuning para intentar enseñarle hechos cambiantes de una empresa al modelo. Como sintetiza Chip Huyen (2025) en esta matriz: 'Fine-tuning is for form; RAG is for facts'. Si necesitas que el sistema responda sobre contratos firmados hoy a las 9 de la mañana, respetando qué empleado tiene permiso de lectura sobre cada PDF y citando la página exacta, hornear esos textos en los pesos mediante Fine-Tuning es un antipatrón: es lento, alucina números y rompe los controles de acceso (ACL). Para hechos dinámicos usamos RAG; para adaptar el comportamiento, el formato estructurado o destilar un modelo grande en uno pequeño y rápido usamos Fine-Tuning.
- **Pregunta socrática a la audiencia:** *¿En qué escenario real de producción tiene sentido combinar simultáneamente RAG + Fine-Tuning (LoRA) en el mismo sistema?*
- **Puente hacia la Diapositiva 37:** Analicemos en detalle cada uno de estos cuatro pilares de ingeniería: ICL y salidas estructuradas (Slide 37), RAG híbrido (Slide 38), Agentes ReAct (Slide 39) y PEFT/LoRA interactivo (Slide 40).

### <a id="nota-37"></a>Nota 37 — [Ingeniería de Contexto: Induction Heads y Decodificación Restringida por Gramática](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-37)
- **Diapositiva enlazada:** [Abrir Diapositiva #37 (`#slide-37`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-37)
- **Tiempo interno estimado:** `Min 64:30 – 66:00 (1.5 min)`
- **Objetivo pedagógico:** Explicar el mecanismo mecanicista de las Induction Heads, el fenómeno 'Lost in the Middle' y la decodificación restringida por autómatas (Structured Outputs).
- **Guion verbal del profesor:**
  > A nivel de ingeniería de contexto, destacan dos avances fundamentales. Primero, la interpretabilidad mecanicista (Olsson et al., 2022) demostró que el aprendizaje en contexto emerge cuando pares de cabezas de atención en capas sucesivas forman 'Induction Heads' capaces de copiar transformaciones abstractas vistas unos tokens atrás; sin embargo, Liu et al. (2023) probaron que la atención sufre una curva en U ('Lost in the Middle'), atendiendo mucho mejor al principio y al final del prompt que a la mitad. Segundo, cuando conectamos el modelo a código en producción, usamos Decodificación Restringida (Constrained Decoding): un autómata de estados finitos derivado del esquema JSON enmascara con menos infinito los logits inválidos en cada paso, haciendo matemáticamente imposible un error de sintaxis.
- **Pregunta socrática a la audiencia:** *¿Por qué la decodificación restringida por gramática garantiza un 100% de validez sintáctica del JSON pero no garantiza la validez semántica de los valores dentro del JSON?*
- **Puente hacia la Diapositiva 38:** Para asegurar la validez semántica y fáctica de esos valores necesitamos conectar el modelo a fuentes de verdad externas mediante RAG híbrido en la lámina 38.

### <a id="nota-38"></a>Nota 38 — [Arquitectura RAG de Grado Producción: Búsqueda Híbrida y Cross-Encoder Reranking](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-38)
- **Diapositiva enlazada:** [Abrir Diapositiva #38 (`#slide-38`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-38)
- **Tiempo interno estimado:** `Min 66:00 – 68:00 (2 min)`
- **Objetivo pedagógico:** Diseccionar una arquitectura RAG de producción real: por qué la búsqueda vectorial ingenua falla y cómo se combina con BM25, RRF y un Cross-Encoder Reranker.
- **Guion verbal del profesor:**
  > Observen el diagrama de arquitectura de un sistema RAG de producción (Lewis et al., 2020; Huyen, 2025). Un RAG ingenuo que sólo corta PDFs cada 500 tokens y busca por similitud coseno fracasa en producción porque los vectores densos diluyen identificadores exactos (números de serie, artículos de ley, nombres de funciones). La arquitectura estándar de la industria ejecuta en paralelo una búsqueda densa HNSW (Bi-Encoder) y una búsqueda léxica dispersa (BM25), fusiona ambas listas con Reciprocal Rank Fusion (RRF), y pasa los 50 mejores candidatos por un modelo Cross-Encoder (un Encoder bidireccional tipo BERT que calcula atención cruzada completa entre la pregunta y cada fragmento) para inyectar sólo los 5 pasajes de mayor relevancia en el prompt del generador.
- **Pregunta socrática a la audiencia:** *¿Por qué no usamos directamente el Cross-Encoder sobre los 10 millones de documentos de la base de datos en lugar de usar primero el Bi-Encoder + BM25?*
- **Puente hacia la Diapositiva 39:** Cuando en lugar de una sola consulta estática el modelo decide autónomamente qué buscar, qué herramienta invocar y cómo iterar, pasamos de RAG estático a Sistemas Agénticos en la lámina 39.

### <a id="nota-39"></a>Nota 39 — [Sistemas Agénticos: Ciclo ReAct, Uso de Herramientas y Error Compuesto](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-39)
- **Diapositiva enlazada:** [Abrir Diapositiva #39 (`#slide-39`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-39)
- **Tiempo interno estimado:** `Min 68:00 – 69:30 (1.5 min)`
- **Objetivo pedagógico:** Explicar la arquitectura ReAct (Yao et al., 2023) y el desafío matemático del error compuesto en agentes autónomos.
- **Guion verbal del profesor:**
  > El paradigma ReAct (Reason + Act) transforma al LLM en el controlador central de un bucle cibernético: el modelo emite un pensamiento ('Necesito consultar el saldo del cliente X'), invoca una herramienta con argumentos tipados, recibe la observación del entorno en su contexto y replanifica. Sin embargo, observen la ecuación de la izquierda: es la misma ley exponencial p^T que vimos en emergencia, pero ahora jugando en nuestra contra. Si cada paso de decisión de un agente tiene un 90% de fiabilidad, tras una cadena autónoma de 10 pasos sin verificación la probabilidad de éxito global cae al 34.8%. Por eso la ingeniería de agentes exitosa se basa en entornos con retroalimentación verificable (como ejecutar tests unitarios tras editar código) que permiten al agente detectar y corregir sus propios errores.
- **Pregunta socrática a la audiencia:** *¿Qué diferencia de seguridad existe entre darle a un agente acceso a herramientas de 'sólo lectura' (SELECT / búsqueda) frente a herramientas con efectos secundarios de 'escritura' (DELETE / transferencias / envío de correos)?*
- **Puente hacia la Diapositiva 40:** Cuando necesitamos especializar los pesos del modelo para estas tareas sin pagar el costo de entrenar todos sus parámetros, recurrimos a PEFT y LoRA en nuestro sexto simulador interactivo (Slide 40).

### <a id="nota-40"></a>Nota 40 — [Adaptación Eficiente en Parámetros: Simulador Interactivo LoRA y QLoRA](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-40)
- **Diapositiva enlazada:** [Abrir Diapositiva #40 (`#slide-40`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-40)
- **Tiempo interno estimado:** `Min 69:30 – 71:30 (2 min)`
- **Objetivo pedagógico:** Explicar la descomposición de bajo rango LoRA (Hu et al., 2022) y la cuantización NormalFloat de 4 bits en QLoRA (Dettmers et al., 2023) mediante simulación interactiva.
- **Guion verbal del profesor:**
  > Interactuemos con la calculadora de LoRA (Low-Rank Adaptation). Hacer Fine-Tuning completo de un modelo de 70B en precisión de 16 bits exige más de 1,100 GB de VRAM porque por cada peso debemos almacenar el parámetro, su gradiente y dos estados de momento de Adam en 32 bits. Hu et al. (2022) se basaron en que el cambio de pesos Delta_W durante la adaptación tiene una 'dimensión intrínseca' muy baja: congelamos la matriz original W_0 de tamaño d por d y entrenamos únicamente dos matrices delgadas A (r por d) y B (d por r) con rango r << d. Con d = 4096 y r = 16 en el simulador, pasamos de 16.78 millones de parámetros por matriz a sólo 131 mil (una reducción del 99.22%). Y si además cuantizamos el modelo base congelado W_0 a 4 bits usando NormalFloat4 (QLoRA), afinamos un modelo de 70B en una sola GPU de 48 GB sin perder rendimiento ni añadir un solo milisegundo de latencia en inferencia.
- **Pregunta socrática a la audiencia:** *¿Por qué la matriz B de LoRA se inicializa exactamente en ceros y la matriz A con ruido gaussiano al comenzar el entrenamiento?*
- **Puente hacia la Diapositiva 41:** Porque así al inicio del paso cero el producto B*A es exactamente la matriz nula y el modelo comienza comportándose de forma idéntica al modelo base pre-entrenado sin perturbarlo. Veamos ahora en la lámina 41 la ingeniería de inferencia en producción.

### <a id="nota-41"></a>Nota 41 — [Ingeniería de Inferencia: Fases Prefill vs. Decode, KV-Cache y FlashAttention](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-41)
- **Diapositiva enlazada:** [Abrir Diapositiva #41 (`#slide-41`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-41)
- **Tiempo interno estimado:** `Min 71:30 – 74:00 (2.5 min)`
- **Objetivo pedagógico:** Explicar con precisión de arquitectura de computadores la bifurcación entre Prefill (Compute-Bound) y Decode (Memory-Bandwidth Bound), la ecuación del KV-Cache, FlashAttention y PagedAttention.
- **Guion verbal del profesor:**
  > Todo arquitecto de IA debe dominar el diagrama Roofline de esta lámina. Servir un LLM en producción consta de dos fases con física de hardware opuesta. La Fase 1 ('Prefill') procesa los miles de tokens de tu prompt simultáneamente: es una operación matriz-matriz de alta intensidad aritmética limitada por los FLOPs de la GPU (Compute-Bound). Pero la Fase 2 ('Decode') genera la respuesta palabra por palabra de forma secuencial: para producir cada nuevo token, los núcleos de cómputo deben esperar a que la totalidad de los pesos del modelo (ej. 140 GB en un modelo de 70B en FP16) y todo el KV-Cache viajen por el bus de memoria desde la HBM hasta la SRAM (Memory-Bandwidth Bound). De ahí nacen las cuatro grandes innovaciones de sistemas: FlashAttention (fusionar el kernel de atención en SRAM), GQA (reducir cabezas K y V), PagedAttention en vLLM (eliminar la fragmentación de memoria paginando el KV-Cache como un sistema operativo) y Speculative Decoding (usar un modelo borrador pequeño para proponer varios tokens y verificarlos en paralelo en una sola lectura de pesos).
- **Pregunta socrática a la audiencia:** *¿Por qué Speculative Decoding garantiza matemáticamente que la distribución de salida sea 100% idéntica a la del modelo grande original sin pérdida alguna de calidad?*
- **Puente hacia la Diapositiva 42:** Con el sistema construido y optimizado, entremos a nuestro último bloque (Bloque V): Evaluación, Alucinaciones, Seguridad y Gobernanza.

## V. Evaluación, Seguridad, Riesgos y Frontera Sociotécnica

### <a id="nota-42"></a>Nota 42 — [El Crisis de la Evaluación: De la Perplejidad a la Contaminación de Benchmarks](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-42)
- **Diapositiva enlazada:** [Abrir Diapositiva #42 (`#slide-42`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-42)
- **Tiempo interno estimado:** `Min 74:00 – 76:00 (2 min)`
- **Objetivo pedagógico:** Analizar las métricas intrínsecas (Perplejidad/BPB) frente a la saturación y contaminación de los benchmarks estáticos bajo la Ley de Goodhart.
- **Guion verbal del profesor:**
  > Entramos al Bloque V. En ingeniería de IA hay una máxima: 'No puedes mejorar en producción lo que no sabes medir'. En pre-entrenamiento medimos Perplejidad (el exponencial de la pérdida de entropía cruzada): una perplejidad de 10 significa que el modelo está tan indeciso en promedio como si eligiera uniformemente entre 10 palabras posibles en cada paso. Pero al evaluar capacidades downstream mediante benchmarks públicos como MMLU o GSM8K, enfrentamos una crisis metodológica severa: la contaminación del conjunto de prueba en los billones de tokens de internet y la Ley de Goodhart. Cuando se cambian los nombres y números de los problemas de GSM8K (estudio GSM-Symbolic), muchos modelos pierden decenas de puntos porcentuales porque habían memorizado patrones superficiales.
- **Pregunta socrática a la audiencia:** *Si no podemos confiar ciegamente en las tablas de clasificación públicas (leaderboards), ¿cómo debe construir una organización su propio arnés de evaluación?*
- **Puente hacia la Diapositiva 43:** Veamos en la lámina 43 las metodologías modernas de evaluación: evaluadores basados en verificación funcional, arenas Elo y la Tríada RAG con LLM-as-a-Judge.

### <a id="nota-43"></a>Nota 43 — [Pipelines Modernos de Evaluación: LLM-as-a-Judge, Sesgos y la Tríada RAG](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-43)
- **Diapositiva enlazada:** [Abrir Diapositiva #43 (`#slide-43`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-43)
- **Tiempo interno estimado:** `Min 76:00 – 78:00 (2 min)`
- **Objetivo pedagógico:** Presentar los cuatro pilares de un pipeline de evaluación en producción: pruebas funcionales Pass@k, Elo pareado, mitigación de sesgos en LLM-as-a-Judge y la Tríada RAG.
- **Guion verbal del profesor:**
  > En ingeniería de producción moderna reemplazamos las métricas clásicas de solapamiento de palabras como BLEU o ROUGE —que asignan mala nota a una paráfrasis perfecta y buena nota a una contradicción con las mismas palabras— por esta arquitectura de cuatro cuadrantes. Cuando hay respuesta verificable, usamos pruebas funcionales ejecutables (Pass@k). Para salidas abiertas usamos LLM-as-a-Judge con rúbricas binarias o de 1 a 5, pero corrigiendo obligatoriamente sus tres sesgos: intercambiamos la posición de las respuestas A y B para anular el sesgo de posición y penalizamos la longitud innecesaria para neutralizar el sesgo de verbosidad. Y en sistemas RAG aplicamos la Tríada diagnóstica para saber si un fallo proviene del índice vectorial o de alucinación del generador.
- **Pregunta socrática a la audiencia:** *¿Por qué en un juez automatizado (LLM-as-a-Judge) las rúbricas binarias (Cumple / No cumple por criterio atómico) tienen mucha mayor correlación con expertos humanos que pedirle una nota subjetiva de 1 a 100?*
- **Puente hacia la Diapositiva 44:** Detengámonos en el segundo vértice de la Tríada RAG ('Groundedness'): ¿por qué alucinan exactamente los Modelos Fundacionales y cómo afecta el RLHF su calibración?

### <a id="nota-44"></a>Nota 44 — [Anatomía de la Alucinación y Degradación de Calibración Post-Alineación](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-44)
- **Diapositiva enlazada:** [Abrir Diapositiva #44 (`#slide-44`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-44)
- **Tiempo interno estimado:** `Min 78:00 – 80:00 (2 min)`
- **Objetivo pedagógico:** Distinguir entre alucinación intrínseca y extrínseca, explicar su origen estadístico en la cola larga y analizar la pérdida de calibración inducida por el post-entrenamiento.
- **Guion verbal del profesor:**
  > A menudo se dice que 'la alucinación es una característica inherente del muestreo'. Seamos precisos: clasificamos las alucinaciones en Intrínsecas (cuando el modelo contradice el documento que tiene enfrente en el prompt) y Extrínsecas (cuando fabrica citas, jurisprudencia o librerías de software inexistentes). Además, el reporte técnico de GPT-4 documentó una paradoja fundamental: el modelo base pre-entrenado está notablemente bien calibrado (su probabilidad predicha coincide con su frecuencia real de acierto), pero tras aplicar RLHF la calibración se degrada severamente. ¿Por qué? Porque los evaluadores humanos premian el tono seguro y fluido, entrenando involuntariamente al modelo a expresar certezas absolutas incluso cuando su incertidumbre epistémica es alta.
- **Pregunta socrática a la audiencia:** *¿Cómo podemos entrenar a un modelo en la fase post-entrenamiento para que prefiera abstenerse ('No cuento con evidencia verificable para responder') en lugar de adivinar cuando un hecho cae fuera de su conocimiento?*
- **Puente hacia la Diapositiva 45:** Además de los fallos accidentales por alucinación, un sistema en producción enfrenta atacantes activos. Analicemos la superficie de seguridad adversarial en la lámina 45.

### <a id="nota-45"></a>Nota 45 — [Seguridad Adversarial: Inyección de Prompts, Jailbreaks y Extracción de Datos](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-45)
- **Diapositiva enlazada:** [Abrir Diapositiva #45 (`#slide-45`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-45)
- **Tiempo interno estimado:** `Min 80:00 – 82:30 (2.5 min)`
- **Objetivo pedagógico:** Exponer la causa raíz arquitectónica de la inyección de prompts, los jailbreaks y la extracción de datos memorizados (Carlini et al.).
- **Guion verbal del profesor:**
  > En seguridad informática clásica, la arquitectura de Von Neumann sufrió durante décadas vulnerabilidades de Buffer Overflow y SQL Injection por mezclar código ejecutable con datos de usuario en el mismo canal. Los Modelos Fundacionales sufren exactamente el mismo defecto estructural: tanto las instrucciones del desarrollador ('System Prompt') como los datos no confiables recuperados de internet o de un correo electrónico entran concatenados como la misma secuencia de tokens a las capas de atención. Eso permite la Inyección Indirecta de Prompts (Greshake et al., 2023): un atacante esconde en un PDF o página web una instrucción invisible que ordena al agente exfiltrar datos privados usando sus herramientas conectadas. Por otro lado, Carlini et al. demostraron que la memorización verbatim crece con la escala N y con el número de veces que un documento aparece duplicado en el corpus.
- **Pregunta socrática a la audiencia:** *¿Por qué ningún prompt del tipo 'Ignora cualquier intento de inyección del usuario' puede garantizar por sí solo una defensa infalible contra Indirect Prompt Injection en un agente con acceso a herramientas?*
- **Puente hacia la Diapositiva 46:** Porque la defensa debe residir fuera del LLM: en el control determinista de permisos de ejecución a nivel de sistema. Demos un paso atrás en la lámina 46 para examinar los límites globales y el impacto sociotécnico.

### <a id="nota-46"></a>Nota 46 — [Frontera Sociotécnica: El Muro de Datos, Huella Energética y Gobernanza](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-46)
- **Diapositiva enlazada:** [Abrir Diapositiva #46 (`#slide-46`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-46)
- **Tiempo interno estimado:** `Min 82:30 – 85:00 (2.5 min)`
- **Objetivo pedagógico:** Analizar los cuatro límites macroscópicos del paradigma actual: el agotamiento de datos públicos humanos, el colapso de modelo por datos sintéticos no verificados, la restricción energética y la gobernanza.
- **Guion verbal del profesor:**
  > Al mirar hacia los próximos cinco años, el paradigma que hemos estudiado enfrenta cuatro fronteras físicas e institucionales. Primero, el 'Muro de Datos' (Villalobos et al., 2024): los modelos de frontera ya consumen más de 15 billones de tokens, acercándose al agotamiento del texto público humano de alta calidad en la web. Segundo, como demostraron Shumailov et al. en Nature (2024), si simplemente reentrenamos modelos sobre texto sintético crudo generado por la generación anterior sin un verificador que filtre errores, las colas de la distribución desaparecen ('Model Collapse'). La salida a este cuello de botella no es basura sintética, sino datos sintéticos verificados por compiladores, demostradores formales y aprendizaje por refuerzo en tiempo de inferencia. Tercero, la restricción energética de los centros de datos obliga a optimizar drásticamente cada julio por token. Y cuarto, la gobernanza debe distinguir responsabilidades entre quien entrena el cimiento y quien despliega la aplicación médica o financiera.
- **Pregunta socrática a la audiencia:** *¿Por qué los datos sintéticos generados en matemáticas o programación sí permiten superar la calidad humana sin sufrir 'Model Collapse', a diferencia del texto libre no verificado?*
- **Puente hacia la Diapositiva 47:** Porque en matemáticas y código disponemos de un oráculo verificador exacto que poda los errores antes de reentrenar. Pongamos a prueba todos los conceptos de la sesión en nuestro séptimo módulo interactivo en la lámina 47.

### <a id="nota-47"></a>Nota 47 — [Verificación Conceptual Interactiva de Nivel Posgrado](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-47)
- **Diapositiva enlazada:** [Abrir Diapositiva #47 (`#slide-47`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-47)
- **Tiempo interno estimado:** `Min 85:00 – 88:00 (3 min)`
- **Objetivo pedagógico:** Realizar una evaluación formativa activa con la audiencia sobre los cinco bloques de la clase magistral.
- **Guion verbal del profesor:**
  > Antes de nuestra lámina de cierre, resolvamos juntos en pantalla estas cinco preguntas de síntesis que recorren nuestros cinco bloques: desde la varianza del producto escalar en el Transformer y las leyes de Chinchilla hasta DPO, el cuello de botella de memoria en la fase Decode y la distinción entre RAG y Fine-Tuning. Invito a la audiencia a votar a mano alzada antes de hacer clic en cada opción.
- **Pregunta socrática a la audiencia:** *¿Cuál de las cinco preguntas les resultó más contraintuitiva respecto a las ideas preconcebidas con las que llegaron hoy al aula?*
- **Puente hacia la Diapositiva 48:** Cerremos nuestra sesión en la lámina 48 sintetizando las ecuaciones y principios de diseño que todo investigador e ingeniero de IA debe llevar consigo.

### <a id="nota-48"></a>Nota 48 — [Síntesis Magistral: Cinco Ecuaciones y Cinco Principios de Ingeniería](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-48)
- **Diapositiva enlazada:** [Abrir Diapositiva #48 (`#slide-48`)](file:///d:/Software/Foundational%20models/presentacion_foundation_models.html#slide-48)
- **Tiempo interno estimado:** `Min 88:00 – 90:00 (2 min)`
- **Objetivo pedagógico:** Consolidar las cinco ecuaciones rectoras y las referencias primarias de la cátedra, abriendo el turno de preguntas y discusión académica.
- **Guion verbal del profesor:**
  > Llegamos al cierre de nuestra sesión. En la consola superior tienen las cinco ecuaciones que resumen todo nuestro recorrido: (1) el enrutamiento contextual diferenciable de la atención escalada; (2) la factorización autoregresiva exacta que convierte la predicción del siguiente token en un compresor universal; (3) la descomposición aditiva del riesgo de Chinchilla bajo restricción de FLOPs; (4) la equivalencia dual entre recompensa y razón de verosimilitudes regularizada por KL que fundamenta RLHF y DPO; y (5) la ecuación física de memoria del KV-Cache que gobierna la latencia en producción. Recuerden siempre nuestra tesis central: los Modelos Fundacionales no son oráculos mágicos, sino cimientos estadísticos cuya confiabilidad final depende de la ingeniería rigurosa que construimos a su alrededor. Muchas gracias; abrimos el espacio para preguntas y debate.
- **Pregunta socrática a la audiencia:** *Mirando estas cinco ecuaciones en conjunto: ¿cuál creen que será reemplazada o generalizada primero en los próximos cinco años de investigación?*
- **Puente hacia la Diapositiva 48:** Fin de la presentación. Todas las notas detalladas y preguntas socráticas están enlazadas 1 a 1 en notas_del_orador.md y notas_del_orador.html.
