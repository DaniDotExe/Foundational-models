# =============================================================================
#   FM GRAND PRIX · Un Modelo Base (Qwen2.5-0.5B), Tres Fases, Tres Vías
#
#   ✏️  ESTE ES EL ÚNICO ARCHIVO QUE DEBES EDITAR (o cópialo desde simulador_y_arena.html).
#   Presupuesto máximo para las 3 fases juntas: 100 CRÉDITOS (basado en FLOPs reales).
# =============================================================================

# >>>>>>>>>>  EDITA DESDE AQUÍ  >>>>>>>>>>

EQUIPO = "Dani"
TEAM_TOKEN = "token_estudiante_01"
GCP_API_URL = "https://shares-loud-latitude-routine.trycloudflare.com/"   # Tu profesor te dará la IP/URL de GCP al iniciar la clase

# --- BLOQUE 1: TUBERÍA DE CURACIÓN DE DATOS (Aplica a los datasets de las 3 fases) ---
CURACION = {
    "dedup_minhash": True,          # True / False: elimina ejemplos casi idénticos con MinHash (3-gramas)
    "umbral_minhash": 0.82,         # [0.50 - 0.98] cuidado: muy bajo (<0.65) borra ejemplos útiles!
    "filtrar_ruido": True,          # True / False: descarta JSON rotos, cuentas contradictorias y choques
    "fraccion_datos_D": 0.80,       # [0.20 - 1.00] fracción de datos limpios (D) que usas para entrenar
}

# --- FASE 1: TABLERO DE SENSORES (Texto ruidoso -> JSON de alerta) ---
FASE_1 = {
    "via": "ICL",                   # "ICL" (Prompting) | "LORA" | "FULL_FT"
    # Si via == "ICL":
    "modo_icl": "few_shot",         # "zero_shot" | "one_shot" | "few_shot"
    "k_ejemplos": 4,                # [1 - 15] cuántos ejemplos meter al prompt
    "usar_cot": False,              # True / False ("Pensemos paso a paso")
    "temperature": 0.0,             # [0.0 - 1.0]
    "max_tokens": 40,               # [16 - 160]
    # Si via == "LORA" o "FULL_FT":
    "lora_r": 8,                    # [2, 4, 8, 16, 32]
    "lora_alpha": 16,               # [4 - 64]
    "epochs": 2,                    # [1 - 8]
    "lr": 3e-4,                     # [1e-5 - 2e-3]
}

# --- FASE 2: CÁLCULO DE FRENADO Y ESQUIVE (Física de 3 pasos -> SI / NO) ---
FASE_2 = {
    "via": "ICL",                   # "ICL" (Prompting) | "LORA" | "FULL_FT"
    # Si via == "ICL":
    "modo_icl": "few_shot",         # "zero_shot" | "one_shot" | "few_shot"
    "k_ejemplos": 3,                # [1 - 15]
    "usar_cot": True,               # True / False ("Pensemos paso a paso" desbloquea la aritmética!)
    "temperature": 0.0,             # [0.0 - 1.0]
    "max_tokens": 110,              # [16 - 160] ojo: CoT necesita ~80-120 tokens para no cortarse!
    # Si via == "LORA" o "FULL_FT":
    "lora_r": 8,
    "lora_alpha": 16,
    "epochs": 3,
    "lr": 3e-4,
}

# --- FASE 3: ¡A LA PISTA! PILOTO AUTÓNOMO (15 tramos leyendo sensores -> Maniobra) ---
FASE_3 = {
    "via": "LORA",                  # "ICL" | "LORA" | "FULL_FT"
    # Si via == "ICL":
    "modo_icl": "few_shot",
    "k_ejemplos": 5,
    "usar_cot": False,
    "temperature": 0.1,
    "max_tokens": 24,
    # Si via == "LORA" o "FULL_FT":
    "lora_r": 8,                    # [2, 4, 8, 16, 32]
    "lora_alpha": 16,
    "epochs": 3,                    # [1 - 8]
    "lr": 5e-4,                     # [1e-5 - 2e-3]
}

# <<<<<<<<<<  EDITA HASTA AQUÍ  <<<<<<<<<<

CONFIG = {
    "equipo": EQUIPO,
    "token": TEAM_TOKEN,
    "gcp_url": GCP_API_URL,
    "curacion": CURACION,
    "fase1": FASE_1,
    "fase2": FASE_2,
    "fase3": FASE_3,
}
