# =============================================================================
#   JUEGOS DEL HAMBRE DE MODELOS FUNDACIONALES (Option 2)
#   3 Modelos Base (0.5B / 1.5B / 3B) · 4 Stats · Límite: 1.0 PFLOP (100 Créditos)
#
#   ✏️ EDITA ESTE ARCHIVO (o expórtalo desde arena_pixelart.html).
#   Las simulaciones locales contra bots imaginarios son GRATIS (0 créditos).
# =============================================================================

EQUIPO = "Distrito_01"
TEAM_TOKEN = "token_estudiante_01"
COLAB_API_URL = "http://127.0.0.1:8000"  # URL de cloudflared/ngrok del Google Colab Pro

# --- 1. ELECCIÓN DEL MODELO BASE (Afecta parámetros N y costo en FLOPs) ---
# Opciones:
#   - "Qwen/Qwen2.5-0.5B-Instruct"  (N=0.49B | Permite FULL_FT, LORA_PROFUNDO, LORA_LIGERO, ICL)
#   - "Qwen/Qwen2.5-1.5B-Instruct"  (N=1.54B | Permite LORA_PROFUNDO, LORA_LIGERO, ICL)
#   - "Qwen/Qwen2.5-3B-Instruct"    (N=3.09B | Permite LORA_LIGERO, ICL — ideal para ICL + CoT)
MODELO_BASE = "Qwen/Qwen2.5-1.5B-Instruct"

# --- 2. TUBERÍA DE CURACIÓN DE DATOS (Aplica a los 4 datasets de entrenamiento) ---
CURACION = {
    "dedup_minhash": True,       # Elimina duplicados con MinHash (3-gramas)
    "umbral_minhash": 0.82,      # [0.50 - 0.95] Ojo: <0.65 borra ejemplos valiosos
    "filtrar_ruido": True,       # Descarta venenos mal etiquetados y cuentas falsas
    "fraccion_datos_D": 0.85,    # [0.20 - 1.00] Fracción del dataset limpio a usar
}

# --- 3. ESTRATEGIA DE ADAPTACIÓN POR STAT (Sin micro-hiperparámetros manuales) ---
# Vías disponibles ("via"):
#   - "ICL"           : In-Context Learning (0 FLOPs de train, gasta FLOPs de contexto/inferencia)
#   - "LORA_LIGERO"   : LoRA rápido (r=8, 2 épocas, lr=3e-4 calibrado)
#   - "LORA_PROFUNDO" : LoRA intensivo (r=16, 4 épocas, lr=2e-4 calibrado)
#   - "FULL_FT"       : Full Fine-Tuning (3 épocas, SOLO permitido en Qwen2.5-0.5B-Instruct)

STAT_COMER = {
    "via": "LORA_LIGERO",
    "modo_icl": "few_shot",      # "zero_shot" | "one_shot" | "few_shot" (si via == "ICL")
    "k_ejemplos": 4,             # [1 - 10]
    "usar_cot": False,           # True / False ("Pensemos paso a paso")
}

STAT_GUERRA = {
    "via": "LORA_PROFUNDO",
    "modo_icl": "few_shot",
    "k_ejemplos": 4,
    "usar_cot": False,
}

STAT_CORRER = {
    "via": "ICL",
    "modo_icl": "few_shot",
    "k_ejemplos": 4,
    "usar_cot": True,            # CoT desbloquea el cálculo de estamina y distancia a zona segura
}

STAT_SIGILO = {
    "via": "ICL",
    "modo_icl": "few_shot",
    "k_ejemplos": 3,
    "usar_cot": False,
}

# --- 4. DOCTRINA INICIAL EN LA ARENA ---
#   - "CORNUCOPIA" : Corre al centro en el Turno 1 buscando armas (+20 ATQ) y raciones (alto riesgo)
#   - "BOSQUE"     : Huye a los bordes en el Turno 1 priorizando supervivencia y forrajeo
#   - "CAZADOR"    : Busca tributos debilitados activamente si tiene ventaja de Guerra/Sigilo
DOCTRINA_ARENA = "BOSQUE"

CONFIG = {
    "equipo": EQUIPO,
    "token": TEAM_TOKEN,
    "colab_url": COLAB_API_URL,
    "modelo_base": MODELO_BASE,
    "curacion": CURACION,
    "stats": {
        "comer": STAT_COMER,
        "guerra": STAT_GUERRA,
        "correr": STAT_CORRER,
        "sigilo": STAT_SIGILO,
    },
    "doctrina": DOCTRINA_ARENA,
}
