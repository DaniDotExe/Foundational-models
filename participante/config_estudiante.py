"""Panel único de hiperparámetros del estudiante — UIS Grand Prix Foundationals Models.

Edita este archivo para experimentar comparando:
1. In-Context Learning (ICL): el modelo base Qwen2.5-1.5B-Instruct NO modifica sus pesos;
   aprende directamente desde ejemplos seleccionados en el prompt (KNN ponderado o aleatorio).
   - Ventaja: iteración instantánea sin entrenamiento.
   - Reto: prompts más largos aumentan la latencia por paso (penalizada en el ranking).

2. Fine-Tuning con LoRA (PEFT): adapta matrices de bajo rango por estudiante.
   - Ventaja: inferencia ultra rápida (zero-shot compacto) y mejor generalización a Pistas B.
   - Reto: calidad de los datos grabados, riesgo de sobreajuste con pocas muestras o muchas épocas.
"""

# Estrategia activa: "ICL", "LORA" o "LORA_ICL"
ESTRATEGIA = "LORA"

# Representación de estado enviada al modelo (configuración óptima)
CONFIG_DATOS = {
    # Número de acciones previas incluidas en H:<historial> (1 a 8 frames)
    "ventana_historia": 5,
    # Resolución de discretización de cada rayo [0..1] -> enteros [0..precision_sensores]
    "precision_sensores": 50,
}

# Hiperparámetros para modo In-Context Learning (ESTRATEGIA in {"ICL", "LORA_ICL"})
CONFIG_ICL = {
    # Cantidad de ejemplos en el prompt (1 a 30)
    "k_shots": 12,
    # Método de selección de ejemplos: "knn" (vecinos más cercanos) o "random"
    "seleccion": "knn",
    # Pesos para [izq, diag_izq, frente, diag_der, der] al calcular distancia KNN
    "pesos_sensores": [1.0, 2.0, 3.0, 2.0, 1.0],
    # Creatividad de muestreo (0.0 = determinista / greedy)
    "temperature": 0.0,
    "top_p": 0.9,
}

# Hiperparámetros para modo PEFT/LoRA (ESTRATEGIA in {"LORA", "LORA_ICL"})
CONFIG_LORA = {
    # Máximo de jugadas tomadas de data/mis_jugadas.json para entrenar (máx 300)
    "num_ejemplos": 200,
    # Rango de las matrices LoRA: 2, 4, 8, 16, 32 (máx 32 según servidor)
    "lora_r": 16,
    # Escala de adaptación (regla clásica: 2 * lora_r)
    "lora_alpha": 32,
    # Capas del Transformer a adaptar (Atención + MLP para mejor convergencia)
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Épocas de entrenamiento (1 a 15 según servidor)
    "epochs": 8,
    # Tasa de aprendizaje AdamW
    "learning_rate": 3e-4,
}
