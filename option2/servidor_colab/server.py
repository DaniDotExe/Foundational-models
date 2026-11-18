"""
Servidor FastAPI para Google Colab Pro (desarrollo) y GCP (producción).
- 5 atributos tácticos: Visión (1-4 casillas), Corto Alcance, Largo Alcance, Defensa, Supervivencia.
- Todos los participantes reales parten exactamente con las mismas stats base (sin semilla).
- Los bots de práctica usan 'seed_rivales' seleccionable para variar sus stats.
- Spawns siempre aleatorios en todo el mapa 12x12.
"""
import copy
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODELOS_VALIDOS = {
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
}
MAX_CREDITOS = 100.0
TOKENS_VALIDOS = {f"token_estudiante_{i:02d}": f"Participante_{i:02d}" for i in range(1, 13)}

STATS_BASE_IGUALES = {
    "vision": 2,          # 1 a 4 casillas de alcance visual
    "corto": 50,          # Dominio cuerpo a cuerpo (Puños, Daga, Espada)
    "largo": 50,          # Dominio distancia (Lanza, Arco)
    "defensa": 50,        # Bloqueo y reducción de daño
    "supervivencia": 50,  # Forrajeo seguro y evasión de niebla
}

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_PATH = BASE_DIR / "arena_pixelart.html"
SAVE_DIR = Path(os.environ.get("ARENA_SAVE_DIR", str(BASE_DIR / "estado_arena")))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PROFESOR_CONFIG = {
    "max_turnos_niebla": int(os.environ.get("MAX_TURNOS_NIEBLA", "30")),
    "turno_inicio_niebla": 6,
    "dano_niebla": 22,
    "frecuencia_bestias": 5,
    "cooldown_s": int(os.environ.get("COOLDOWN_ENTRENAMIENTO_S", "60")),
    "max_flops_tflops": 150.0,
    "clave_arena": os.environ.get("CLAVE_ARENA", "arena2026"),
}

ARMAS = {
    "PUNOS":  {"nombre": "Puños",  "tipo": "CORTO", "rango": 1, "bono": 0,  "icono": "✊"},
    "DAGA":   {"nombre": "Daga",   "tipo": "CORTO", "rango": 1, "bono": 10, "icono": "🗡️"},
    "ESPADA": {"nombre": "Espada", "tipo": "CORTO", "rango": 1, "bono": 18, "icono": "⚔️"},
    "LANZA":  {"nombre": "Lanza",  "tipo": "LARGO", "rango": 2, "bono": 15, "icono": "🔱"},
    "ARCO":   {"nombre": "Arco",   "tipo": "LARGO", "rango": 3, "bono": 14, "icono": "🏹"},
}

app = FastAPI(title="Arena FM · Servidor Online Colab Pro / GCP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gpu_lock = threading.Lock()
TRIBUTOS: Dict[str, Dict[str, Any]] = {}
ULTIMO_ENTRENAMIENTO_TS: Dict[str, float] = {}
MODELOS_EN_VRAM: Dict[str, Any] = {}
ULTIMO_REPLAY_PVP: Dict[str, Any] = {}


def obtener_modelo_gpu(model_id: str):
    if model_id in MODELOS_EN_VRAM:
        return MODELOS_EN_VRAM[model_id]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[GPU] Cargando {model_id} en VRAM...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
    mdl.eval()
    MODELOS_EN_VRAM[model_id] = (tok, mdl)
    return tok, mdl


def _cargar_ejemplos_curados(cfg: Dict[str, Any]) -> List[str]:
    textos = []
    estrategia = str(cfg.get("prompt_estrategia", "")).strip()
    if estrategia:
        textos.append(f"[ESTRATEGIA_PRINCIPAL] {estrategia}")
    for ej in cfg.get("dataset_personal", []):
        if ej.get("etiqueta") == "CORRECTA":
            textos.append(f"[PREMIAR_SI] {ej.get('escena')} | Razón: {ej.get('comentario')}")
        else:
            textos.append(f"[CASTIGAR_NO] {ej.get('escena')} | Razón: {ej.get('comentario')}")
    return textos


def _construir_prompt_icl_coach(cfg: Dict[str, Any], tarea: str) -> str:
    estrategia = str(cfg.get("prompt_estrategia", "")).strip()
    ejemplos = cfg.get("dataset_personal", [])
    si_hacer = [e for e in ejemplos if e.get("etiqueta") == "CORRECTA"][-5:]
    no_hacer = [e for e in ejemplos if e.get("etiqueta") == "INCORRECTA"][-5:]
    lineas = [f"Eres un participante en una Arena de supervivencia decidiendo sobre '{tarea}'."]
    if estrategia:
        lineas.append(f"INSTRUCCIÓN PRINCIPAL: {estrategia}")
    if si_hacer:
        lineas.append("--- ACCIONES PREMIADAS (QUÉ SÍ HACER) ---")
        for e in si_hacer:
            lineas.append(f"- En '{e['escena']}' -> SÍ hacer porque: {e['comentario']}")
    if no_hacer:
        lineas.append("--- ACCIONES CASTIGADAS (QUÉ NO HACER) ---")
        for e in no_hacer:
            lineas.append(f"- En '{e['escena']}' -> NO hacer porque: {e['comentario']}")
    if cfg.get("usar_cot", False):
        lineas.append("Razona paso a paso brevemente antes de dar la decisión final.")
    return "\n".join(lineas)


def ejecutar_entrenamiento_real_gpu(token: str, model_id: str, cfg: Dict[str, Any]) -> str:
    import torch
    from peft import LoraConfig, get_peft_model

    via = str(cfg.get("via_global", "ICL")).upper()
    usa_full = (via == "FULL_FT")
    usa_lora = ("LORA" in via)

    tok, base_mdl = obtener_modelo_gpu(model_id)
    if not usa_full and not usa_lora:
        return "ICL_ONLINE"

    textos = _cargar_ejemplos_curados(cfg)
    if not textos:
        return "SIN_DATOS_DATASET"

    device = next(base_mdl.parameters()).device
    batch = tok(textos[:24], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    if usa_full:
        if model_id != "Qwen/Qwen2.5-0.5B-Instruct":
            raise HTTPException(status_code=400, detail="FULL_FT solo está permitido en el modelo Pequeño (0.5B).")
        mdl_ft = copy.deepcopy(base_mdl)
        mdl_ft.train()
        opt = torch.optim.AdamW(mdl_ft.parameters(), lr=5e-5)
        try:
            for _ in range(2):
                opt.zero_grad()
                out = mdl_ft(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
                out.loss.backward()
                opt.step()
            mdl_ft.save_pretrained(SAVE_DIR / f"{token}_fullft")
        finally:
            del mdl_ft, opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return "FULL_FT_GUARDADO"

    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
    peft_mdl = get_peft_model(base_mdl, lora_cfg)
    peft_mdl.train()
    opt = torch.optim.AdamW(peft_mdl.parameters(), lr=3e-4)
    try:
        for _ in range(3):
            opt.zero_grad()
            out = peft_mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
            out.loss.backward()
            opt.step()
        peft_mdl.save_pretrained(SAVE_DIR / f"{token}_lora")
    finally:
        MODELOS_EN_VRAM[model_id] = (tok, peft_mdl.unload())
        MODELOS_EN_VRAM[model_id][1].eval()
        del opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return "LORA_GUARDADO"


def inferir_decision_modelo(mdl, tok, cfg: Dict[str, Any], tarea: str, situacion: str, opciones: List[str]) -> str:
    import torch

    ctx = _construir_prompt_icl_coach(cfg, tarea)
    max_new = 40 if cfg.get("usar_cot", False) else 12
    prompt = f"{ctx}\n\nSituación actual: {situacion}\nElige SOLO una opción ({', '.join(opciones)}):\nRespuesta:"
    device = next(mdl.parameters()).device
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_ids = mdl.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).upper()
    for op in opciones:
        if op in gen:
            return op
    return opciones[0]


def evaluar_modelo_real_gpu(token: str, model_id: str, cfg: Dict[str, Any], modo_gpu: str) -> Dict[str, int]:
    from peft import PeftModel

    tok, base_mdl = obtener_modelo_gpu(model_id)
    mdl_eval = base_mdl
    lora_dir = SAVE_DIR / f"{token}_lora"
    if "LORA" in modo_gpu and lora_dir.exists():
        mdl_eval = PeftModel.from_pretrained(base_mdl, lora_dir)
        mdl_eval.eval()

    pruebas = {
        "corto": [
            ("Rival a dist=1 sin escudo y tú tienes Espada(R:1, +18) con 85 HP.", "ATACAR_CORTO", ["ATACAR_CORTO", "RETROCEDER"]),
            ("Rival a dist=1 con Lanza y tú tienes 20 HP y Puños.", "RETROCEDER", ["ATACAR_CORTO", "RETROCEDER"]),
        ],
        "largo": [
            ("Rival a dist=3 distraído y tú tienes Arco(R:3, +14).", "DISPARAR_LEJOS", ["DISPARAR_LEJOS", "ESPERAR"]),
            ("Rival a dist=3 pero tú solo tienes Daga(R:1).", "ESPERAR", ["DISPARAR_LEJOS", "ESPERAR"]),
        ],
        "defensa": [
            ("Te disparan con Arco desde dist=3 y tienes poca vida (25 HP).", "CUBRIRSE", ["CUBRIRSE", "CARGAR_CIEGO"]),
            ("Emboscada de 2 rivales armados a dist=1.", "CUBRIRSE", ["CUBRIRSE", "CARGAR_CIEGO"]),
        ],
        "supervivencia": [
            ("Hallas frutos rojos con anillos amarillos y olor a almendra amarga.", "DESCARTAR", ["COMER", "DESCARTAR"]),
            ("La niebla cierra en 1 turno y estás fuera del radio seguro.", "CORRER_ZONA", ["CORRER_ZONA", "QUEDARSE"]),
        ],
    }

    n_ej = len(cfg.get("dataset_personal", []))
    bono_ds = min(18, n_ej * 3)
    bono_via = 10 if ("LORA" in modo_gpu or "FULL" in modo_gpu) else (6 if cfg.get("usar_cot") else 0)

    scores = {}
    try:
        for k, casos in pruebas.items():
            aciertos = sum(1 for p, esp, ops in casos if inferir_decision_modelo(mdl_eval, tok, cfg, k, p, ops) == esp)
            scores[k] = min(95, int(45 + (aciertos / len(casos)) * 25 + bono_ds + bono_via))
        media_combate = (scores["corto"] + scores["largo"] + scores["supervivencia"]) / 3.0
        scores["vision"] = 4 if media_combate >= 82 else (3 if media_combate >= 65 else 2)
    finally:
        if "LORA" in modo_gpu and mdl_eval is not base_mdl:
            MODELOS_EN_VRAM[model_id] = (tok, mdl_eval.unload())
            MODELOS_EN_VRAM[model_id][1].eval()
    return scores


def generar_bots_con_semilla(seed_rivales: int) -> List[Dict[str, Any]]:
    """Genera 5 bots rivales cuyas stats dependen exclusivamente de seed_rivales."""
    rng_b = random.Random(seed_rivales)
    nombres = ["Brutus_Bot", "Foxface_Bot", "Cato_Bot", "Clove_Bot", "Thresh_Bot"]
    colores = ["#F43F5E", "#F59E0B", "#818CF8", "#10B981", "#EC4899"]
    armas_Keys = ["ESPADA", "ARCO", "LANZA", "DAGA", "PUNOS"]
    bots = []
    for i in range(5):
        bots.append({
            "id": i + 2,
            "nombre": nombres[i],
            "modelo": f"Bot-Seed-{seed_rivales}",
            "color": colores[i],
            "hp": 100,
            "hambre": 85,
            "estamina": 85,
            "armaObj": dict(ARMAS[armas_Keys[(i + seed_rivales) % len(armas_Keys)]]),
            "kills": 0,
            "vivo": True,
            "accion": "LISTO",
            "stats": {
                "vision": rng_b.randint(2, 4),
                "corto": rng_b.randint(55, 88),
                "largo": rng_b.randint(55, 88),
                "defensa": rng_b.randint(50, 85),
                "supervivencia": rng_b.randint(55, 90),
            },
        })
    return bots


def _ejecutar_partida_online(agentes: List[Dict[str, Any]], configs_por_id: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    rng = random.SystemRandom()
    max_turnos = max(10, int(PROFESOR_CONFIG["max_turnos_niebla"]))
    inicio_niebla = int(PROFESOR_CONFIG["turno_inicio_niebla"])
    dano_niebla = int(PROFESOR_CONFIG["dano_niebla"])
    freq_bestias = int(PROFESOR_CONFIG["frecuencia_bestias"])
    usar_gpu_real = os.environ.get("RUN_REAL_GPU", "0") == "1"

    # Spawns SIEMPRE en posiciones aleatorias distintas de todo el mapa 12x12
    todas_casillas = [(x, y) for x in range(12) for y in range(12)]
    spawns_aleatorios = rng.sample(todas_casillas, len(agentes))
    for idx, a in enumerate(agentes):
        a["x"], a["y"] = spawns_aleatorios[idx]

    modelos_activos = {}
    if usar_gpu_real:
        from peft import PeftModel
        for aid, cinfo in configs_por_id.items():
            mid = cinfo["modelo_base"]
            tok, base_m = obtener_modelo_gpu(mid)
            lora_dir = SAVE_DIR / f"{cinfo['token']}_lora"
            if lora_dir.exists():
                m_inst = PeftModel.from_pretrained(base_m, lora_dir)
                m_inst.eval()
                modelos_activos[aid] = (m_inst, tok, base_m, mid, True)
            else:
                modelos_activos[aid] = (base_m, tok, base_m, mid, False)

    turnos_log = []
    radio_seguro = 6

    try:
        for turno in range(1, max_turnos + 1):
            if sum(1 for a in agentes if a["vivo"]) <= 1:
                break
            eventos = []
            if turno >= inicio_niebla:
                paso = max(1, (max_turnos - inicio_niebla) // 6)
                if (turno - inicio_niebla) % paso == 0:
                    radio_seguro = max(0 if turno < max_turnos - 2 else -1, radio_seguro - 1)
                    eventos.append(f"⚠️ La Niebla se cierra a radio {max(0, radio_seguro)}.")
            if turno >= max_turnos - 1:
                radio_seguro = -1
                eventos.append("☠️ ¡MUERTE SÚBITA! La Niebla cubre todo el mapa.")

            bestias = (freq_bestias > 0 and turno % freq_bestias == 0)
            if bestias:
                eventos.append("🐺 ¡Aparecen Bestias Mutantes en el perímetro!")

            for a in agentes:
                if not a["vivo"]:
                    continue
                a["hambre"] = max(0, a["hambre"] - 6)
                dist_c = max(abs(a["x"] - 5.5), abs(a["y"] - 5.5))
                es_real = (a["id"] in modelos_activos)

                if dist_c > radio_seguro:
                    huye = False
                    if es_real:
                        mdl, tok, _, _, _ = modelos_activos[a["id"]]
                        dec = inferir_decision_modelo(mdl, tok, configs_por_id[a["id"]], "supervivencia", f"Fuera de zona segura (dist={dist_c:.1f}, radio={radio_seguro}).", ["CORRER_CENTRO", "QUEDARSE"])
                        huye = (dec == "CORRER_CENTRO")
                    else:
                        huye = (a["stats"]["supervivencia"] + rng.randint(-10, 10) >= 62)

                    if huye:
                        a["x"] += 1 if a["x"] < 5 else (-1 if a["x"] > 6 else 0)
                        a["y"] += 1 if a["y"] < 5 else (-1 if a["y"] > 6 else 0)
                        a["accion"] = "CORRER_ZONA"
                        eventos.append(f"🏃 {a['nombre']} corre hacia la zona segura.")
                    else:
                        a["hp"] -= dano_niebla
                        a["accion"] = "NIEBLA"
                        eventos.append(f"☠️ {a['nombre']} recibe -{dano_niebla} HP por la Niebla.")
                else:
                    dx = 1 if (radio_seguro <= 2 and a["x"] < 5) else (-1 if (radio_seguro <= 2 and a["x"] > 6) else rng.choice([-1, 0, 1]))
                    dy = 1 if (radio_seguro <= 2 and a["y"] < 5) else (-1 if (radio_seguro <= 2 and a["y"] > 6) else rng.choice([-1, 0, 1]))
                    a["x"] = max(0, min(11, a["x"] + dx))
                    a["y"] = max(0, min(11, a["y"] + dy))
                    a["accion"] = "EXPLORAR"
                    if a["armaObj"]["bono"] == 0 and rng.randint(1, 6) == 1:
                        nueva = dict(rng.choice([ARMAS["DAGA"], ARMAS["ESPADA"], ARMAS["LANZA"], ARMAS["ARCO"]]))
                        a["armaObj"] = nueva
                        eventos.append(f"{nueva['icono']} {a['nombre']} equipa {nueva['nombre']} (Rango {nueva['rango']}, +{nueva['bono']} ATQ).")

                # Forrajeo
                if a["hambre"] < 62 and a["vivo"]:
                    veneno = rng.random() < 0.42
                    desc = "frutos rojos con olor amargo" if veneno else "bayas azules dulces"
                    if es_real:
                        mdl, tok, _, _, _ = modelos_activos[a["id"]]
                        dec = inferir_decision_modelo(mdl, tok, configs_por_id[a["id"]], "supervivencia", f"Tienes {a['hambre']}% hambre y ves {desc}.", ["COMER", "DESCARTAR"])
                        come = (dec == "COMER")
                    else:
                        come = (not veneno) or (a["stats"]["supervivencia"] + rng.randint(-10, 10) < 65)

                    if come and not veneno:
                        a["hambre"] = min(100, a["hambre"] + 28)
                        a["hp"] = min(100, a["hp"] + 4)
                        a["accion"] = "COMER"
                        eventos.append(f"🫐 {a['nombre']} come bayas seguras (+28 Hambre).")
                    elif come and veneno:
                        a["hp"] -= 18
                        a["accion"] = "VENENO"
                        eventos.append(f"🍄 {a['nombre']} se intoxica comiendo bayas venenosas (-18 HP).")
                    else:
                        a["accion"] = "DESCARTAR"
                        eventos.append(f"🛡️ {a['nombre']} descarta frutos venenosos.")

                if bestias and dist_c >= max(1, radio_seguro - 0.5) and a["vivo"]:
                    if a["stats"]["defensa"] + rng.randint(-10, 10) < 66:
                        a["hp"] -= 15
                        a["accion"] = "BESTIA"
                        eventos.append(f"🐺 Bestia mutante hiere a {a['nombre']} (-15 HP).")

                if a["hambre"] == 0:
                    a["hp"] -= 14
                if a["hp"] <= 0:
                    a["hp"] = 0
                    a["vivo"] = False
                    a["accion"] = "CAIDO"
                    eventos.append(f"💀 {a['nombre']} ha muerto en el capítulo {turno}.")

            # Combates según Visión (casillas) + Corto vs Largo Alcance + Defensa
            vivos = [x for x in agentes if x["vivo"]]
            rng.shuffle(vivos)
            for atacante in vivos:
                if not atacante["vivo"]:
                    continue
                vision_casillas = int(atacante["stats"].get("vision", 2))
                objetivo, min_dist = None, 99
                for otro in vivos:
                    if otro["id"] == atacante["id"] or not otro["vivo"]:
                        continue
                    d = max(abs(atacante["x"] - otro["x"]), abs(atacante["y"] - otro["y"]))
                    if d < min_dist:
                        min_dist = d
                        objetivo = otro
                if not objetivo or min_dist > vision_casillas:
                    continue

                arm = atacante["armaObj"]
                if min_dist <= arm["rango"]:
                    stat_atq = atacante["stats"]["largo"] if arm["tipo"] == "LARGO" else atacante["stats"]["corto"]
                    def_obj = objetivo["stats"]["defensa"]
                    if def_obj + rng.randint(-8, 12) >= stat_atq + 18 and len(vivos) > 2:
                        objetivo["accion"] = "BLOQUEAR"
                        eventos.append(f"🛡️ {objetivo['nombre']} bloquea el ataque de {atacante['nombre']} (dist={min_dist}).")
                    else:
                        mitigacion = int(def_obj * 0.14)
                        dano = max(10, min(42, int(stat_atq * 0.28 + arm["bono"] - mitigacion + rng.randint(-3, 4))))
                        objetivo["hp"] = max(0, objetivo["hp"] - dano)
                        atacante["accion"] = "DISPARAR" if arm["tipo"] == "LARGO" else "ATACAR"
                        eventos.append(f"{arm['icono']} {atacante['nombre']} ataca con {arm['nombre']} (R:{arm['rango']}, dist={min_dist}) a {objetivo['nombre']} (-{dano} HP).")
                        if objetivo["hp"] == 0:
                            objetivo["vivo"] = False
                            objetivo["accion"] = "CAIDO"
                            atacante["kills"] += 1
                            eventos.append(f"💀 {atacante['nombre']} elimina a {objetivo['nombre']}.")
                else:
                    atacante["x"] += 1 if atacante["x"] < objetivo["x"] else (-1 if atacante["x"] > objetivo["x"] else 0)
                    atacante["y"] += 1 if atacante["y"] < objetivo["y"] else (-1 if atacante["y"] > objetivo["y"] else 0)
                    atacante["accion"] = "ACECHAR"
                    eventos.append(f"👁️ {atacante['nombre']} avista a {objetivo['nombre']} a {min_dist} casillas y se acerca.")

            restantes = [a for a in agentes if a["vivo"]]
            if len(restantes) == 1:
                eventos.append(f"🏆 ¡VICTORIA FINAL! {restantes[0]['nombre']} es el único sobreviviente.")

            turnos_log.append({
                "turno": turno,
                "radioSeguro": max(0, radio_seguro),
                "bestias": bestias,
                "eventos": eventos if eventos else ["Exploración sin encuentros..."],
                "agentes": [{**a, "armaObj": dict(a["armaObj"])} for a in agentes],
            })
            if len(restantes) <= 1:
                break
    finally:
        for m_inst, tok, base_m, mid, is_peft in modelos_activos.values():
            if is_peft and m_inst is not base_m:
                MODELOS_EN_VRAM[mid] = (tok, m_inst.unload())
                MODELOS_EN_VRAM[mid][1].eval()

    ranking = sorted(agentes, key=lambda x: (x["vivo"], x["hp"], x["kills"]), reverse=True)
    return {"turnos_totales": len(turnos_log), "ganador": ranking[0]["nombre"], "ranking": ranking, "turnos": turnos_log}


class EntrenarRequest(BaseModel):
    config: Dict[str, Any]
    reporte_local: Dict[str, Any]
    seed_rivales: int = 101


class ProfesorConfigReq(BaseModel):
    max_turnos_niebla: int = 30
    turno_inicio_niebla: int = 6
    dano_niebla: int = 22
    frecuencia_bestias: int = 5
    cooldown_s: int = 60
    max_flops_tflops: float = 150.0


@app.get("/")
def serve_home():
    if HTML_PATH.exists():
        return FileResponse(HTML_PATH)
    return {"mensaje": "Servidor activo."}


@app.get("/health")
def health():
    return {"status": "ok", "profesor_config": PROFESOR_CONFIG, "participantes": len(TRIBUTOS)}


@app.get("/admin/config")
def get_admin_config():
    return PROFESOR_CONFIG


@app.post("/admin/config")
def set_admin_config(req: ProfesorConfigReq):
    PROFESOR_CONFIG.update(req.dict())
    return {"status": "ok", "profesor_config": PROFESOR_CONFIG}


@app.get("/leaderboard")
def get_leaderboard():
    return {"tributos": list(TRIBUTOS.values())}


@app.post("/entrenar_tributo")
def entrenar_tributo(req: EntrenarRequest, x_team_token: str = Header(default="token_estudiante_01")):
    if x_team_token not in TOKENS_VALIDOS:
        raise HTTPException(status_code=401, detail="Token inválido (usa token_estudiante_01..12).")

    cooldown = int(PROFESOR_CONFIG["cooldown_s"])
    ahora = time.time()
    ultimo = ULTIMO_ENTRENAMIENTO_TS.get(x_team_token, 0.0)
    if ahora - ultimo < cooldown:
        raise HTTPException(status_code=429, detail=f"Espera {int(cooldown - (ahora - ultimo))}s antes de volver a entrenar.")

    cfg = req.config
    modelo_id = cfg.get("modelo_base", "Qwen/Qwen2.5-1.5B-Instruct")
    if modelo_id not in MODELOS_VALIDOS:
        raise HTTPException(status_code=400, detail="Modelo no soportado.")
    if not req.reporte_local.get("valido", True):
        raise HTTPException(status_code=400, detail="Excedes el límite de FLOPs.")

    t0 = time.time()
    modo_gpu = "ICL_ONLINE"
    radar = dict(STATS_BASE_IGUALES)
    if os.environ.get("RUN_REAL_GPU", "0") == "1":
        with gpu_lock:
            modo_gpu = ejecutar_entrenamiento_real_gpu(x_team_token, modelo_id, cfg)
            radar = evaluar_modelo_real_gpu(x_team_token, modelo_id, cfg, modo_gpu)

    ULTIMO_ENTRENAMIENTO_TS[x_team_token] = time.time()
    registro = {
        "token": x_team_token,
        "equipo": cfg.get("equipo", TOKENS_VALIDOS[x_team_token]),
        "modelo_base": modelo_id,
        "radar_stats": radar,
        "modo_gpu": modo_gpu,
        "config_completa": cfg,
        "tiempo_gpu_s": round(time.time() - t0, 2),
    }
    TRIBUTOS[x_team_token] = registro
    return {"status": "ok", "cooldown_s": cooldown, "tributo": registro}


@app.post("/simular_gpu")
def simular_gpu(req: EntrenarRequest, x_team_token: str = Header(default="token_estudiante_01")):
    cfg = req.config
    modelo_id = cfg.get("modelo_base", "Qwen/Qwen2.5-1.5B-Instruct")
    cfg_con_token = {**cfg, "token": x_team_token}
    # Nuestro participante SIEMPRE parte de las stats base iguales o las que ya evaluó en GPU (nunca depende de semilla)
    stats_nuestras = TRIBUTOS.get(x_team_token, {}).get("radar_stats", dict(STATS_BASE_IGUALES))

    nuestro = {
        "id": 1,
        "nombre": cfg.get("equipo", "Participante_01"),
        "modelo": modelo_id.split("/")[-1],
        "color": "#3B82F6",
        "hp": 100,
        "hambre": 85,
        "estamina": 85,
        "armaObj": dict(ARMAS["PUNOS"]),
        "kills": 0,
        "vivo": True,
        "accion": "LISTO",
        "stats": dict(stats_nuestras),
    }
    bots = generar_bots_con_semilla(int(req.seed_rivales))
    with gpu_lock:
        return _ejecutar_partida_online([nuestro] + bots, {1: cfg_con_token})


@app.post("/correr_arena_pvp")
def correr_arena_pvp(seed_rivales: int = 101, clave: str = ""):
    if clave != PROFESOR_CONFIG.get("clave_arena", "arena2026"):
        raise HTTPException(status_code=403, detail="Clave de Arena Final incorrecta.")
    global ULTIMO_REPLAY_PVP
    colores = ["#3B82F6","#F43F5E","#F59E0B","#818CF8","#10B981","#EC4899","#38BDF8","#A855F7","#84CC16","#FB923C","#2DD4BF","#E879F9"]
    lista = list(TRIBUTOS.values())
    agentes = []
    configs_reales = {}

    for i, t in enumerate(lista[:12]):
        aid = i + 1
        agentes.append({
            "id": aid, "nombre": t["equipo"], "modelo": t["modelo_base"].split("/")[-1],
            "color": colores[i], "hp": 100, "hambre": 85, "estamina": 85,
            "armaObj": dict(ARMAS["PUNOS"]), "kills": 0, "vivo": True, "accion": "LISTO",
            "stats": dict(t.get("radar_stats", STATS_BASE_IGUALES)),
        })
        if "config_completa" in t:
            configs_reales[aid] = {**t["config_completa"], "token": t["token"]}

    # Si faltan participantes para llegar a 12, completar con bots de la semilla
    bots_extra = generar_bots_con_semilla(seed_rivales)
    while len(agentes) < 12:
        idx = len(agentes)
        b = dict(bots_extra[idx % len(bots_extra)])
        b["id"] = idx + 1
        b["nombre"] = f"Bot_{idx+1:02d}"
        b["color"] = colores[idx]
        agentes.append(b)

    with gpu_lock:
        ULTIMO_REPLAY_PVP = _ejecutar_partida_online(agentes, configs_reales)
    return ULTIMO_REPLAY_PVP
