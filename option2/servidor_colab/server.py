"""
Servidor FastAPI para Google Colab Pro (desarrollo) y GCP (producción).
Reglas Oficiales (Múltiplos de 10 · Mapa 12x12 · Visión 2 · Daño a dist <= 1):
- Stats: Vida (100 HP) y Energía (100). Turno normal: -10 Energía; HUIR: -20 Energía; Energía=0 -> -10 HP.
- Armas (solo cuerpo a cuerpo dist<=1): Desarmado (10), Cuchillo (20), Espada (30).
- Objetos (se consumen al usarse): Comida (+40 Energía), Veneno (-20 HP), Botiquín (+30 HP).
- DEFENDER: Si lo atacan varios, bloquea el 100% del golpe más fuerte (y contraataca 10 HP a ese atacante),
  pero recibe completos los ataques adicionales por el flanco.
- Niebla: Cierra hacia una Zona Segura aleatoria (zona_x, zona_y) quitando -20 HP por turno fuera.
"""
import copy
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

MODELOS_VALIDOS = {
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
}
TOKENS_VALIDOS = {f"token_estudiante_{i:02d}": f"Participante_{i:02d}" for i in range(1, 13)}

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_PATH = BASE_DIR / "arena_pixelart.html"
SAVE_DIR = Path(os.environ.get("ARENA_SAVE_DIR", str(BASE_DIR / "estado_arena")))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PROFESOR_CONFIG = {
    "max_turnos_niebla": int(os.environ.get("MAX_TURNOS_NIEBLA", "30")),
    "turno_inicio_niebla": 5,
    "dano_niebla": 20,
    "cooldown_s": int(os.environ.get("COOLDOWN_ENTRENAMIENTO_S", "60")),
    "max_flops_tflops": 150.0,
    "clave_arena": os.environ.get("CLAVE_ARENA", "arena2026"),
}

ARMAS = {
    "DESARMADO": {"nombre": "Desarmado", "dano": 10, "icono": "✊"},
    "CUCHILLO":  {"nombre": "Cuchillo",  "dano": 20, "icono": "🗡️"},
    "ESPADA":    {"nombre": "Espada",    "dano": 30, "icono": "⚔️"},
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
        textos.append(f"[INSTRUCCION_PRINCIPAL] {estrategia}")
    for ej in cfg.get("dataset_personal", []):
        if ej.get("etiqueta") == "CORRECTA":
            textos.append(f"[PREMIAR_SI] {ej.get('escena')} -> Acción: {ej.get('accion_tomada')} | Razón: {ej.get('comentario')}")
        else:
            textos.append(f"[CASTIGAR_NO] {ej.get('escena')} -> Evitar: {ej.get('accion_tomada')} | Razón: {ej.get('comentario')}")
    return textos


def _construir_prompt_icl_coach(cfg: Dict[str, Any]) -> str:
    estrategia = str(cfg.get("prompt_estrategia", "")).strip()
    ejemplos = cfg.get("dataset_personal", [])
    si_hacer = [e for e in ejemplos if e.get("etiqueta") == "CORRECTA"][-5:]
    no_hacer = [e for e in ejemplos if e.get("etiqueta") == "INCORRECTA"][-5:]
    lineas = ["Eres un participante en una Arena de supervivencia 12x12."]
    if estrategia:
        lineas.append(f"TU ESTRATEGIA PRINCIPAL: {estrategia}")
    if si_hacer:
        lineas.append("--- QUÉ SÍ HACER (Ejemplos premiados) ---")
        for e in si_hacer:
            lineas.append(f"- {e['escena']} -> SÍ elegir {e['accion_tomada']}: {e['comentario']}")
    if no_hacer:
        lineas.append("--- QUÉ NO HACER (Errores castigados) ---")
        for e in no_hacer:
            lineas.append(f"- {e['escena']} -> NO elegir {e['accion_tomada']}: {e['comentario']}")
    if cfg.get("usar_cot", False):
        lineas.append("Razona paso a paso brevemente antes de escribir la acción final.")
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


def inferir_accion_json(mdl, tok, cfg: Dict[str, Any], entorno_json: Dict[str, Any]) -> str:
    import torch

    opciones = entorno_json["acciones_validas"]
    ctx = _construir_prompt_icl_coach(cfg)
    max_new = 42 if cfg.get("usar_cot", False) else 12
    prompt = f"{ctx}\n\nEntorno JSON:\n{json.dumps(entorno_json, ensure_ascii=False)}\nElige SOLO una acción de {opciones}:\nAcción:"
    device = next(mdl.parameters()).device
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_ids = mdl.generate(**inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(out_ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).upper()
    for op in opciones:
        if op in gen:
            return op
    return opciones[0]


def _direccion_cardinal(x0: int, y0: int, x1: int, y1: int) -> str:
    ns = "NORTE" if y1 < y0 else ("SUR" if y1 > y0 else "")
    eo = "OESTE" if x1 < x0 else ("ESTE" if x1 > x0 else "")
    return (ns + eo) or "AQUI"


def _paso_hacia(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int]:
    dx = 1 if x1 > x0 else (-1 if x1 < x0 else 0)
    dy = 1 if y1 > y0 else (-1 if y1 < y0 else 0)
    return max(0, min(11, x0 + dx)), max(0, min(11, y0 + dy))


def generar_mapa_objetos(rng: random.Random) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Coloca comida buena (+40 Energía), veneno (-20 HP), botiquines (+30 HP) y armas (Cuchillo 20, Espada 30)."""
    casillas = [(x, y) for x in range(12) for y in range(12)]
    seleccion = rng.sample(casillas, 34)
    objetos = {}
    for i, pos in enumerate(seleccion):
        if i < 10:
            objetos[pos] = {"tipo": "COMIDA", "es_veneno": False, "desc": "Fruta azul de olor dulce (+40 Energía)"}
        elif i < 18:
            objetos[pos] = {"tipo": "COMIDA", "es_veneno": True, "desc": "Hongos rojos con manchas blancas y olor amargo"}
        elif i < 24:
            objetos[pos] = {"tipo": "BOTIQUIN", "desc": "Botiquín médico (+30 Vida)"}
        elif i < 30:
            objetos[pos] = {"tipo": "ARMA", "arma": dict(ARMAS["CUCHILLO"]), "desc": "Cuchillo (20 daño)"}
        else:
            objetos[pos] = {"tipo": "ARMA", "arma": dict(ARMAS["ESPADA"]), "desc": "Espada (30 daño)"}
    return objetos


def generar_bots_con_semilla(seed_rivales: int) -> List[Dict[str, Any]]:
    rng_b = random.Random(seed_rivales)
    nombres = ["Brutus_Bot", "Foxface_Bot", "Cato_Bot", "Clove_Bot", "Thresh_Bot"]
    colores = ["#F43F5E", "#F59E0B", "#818CF8", "#10B981", "#EC4899"]
    perfiles = ["AGRESIVO", "CAUTELOSO", "AGRESIVO", "EQUILIBRADO", "DEFENSIVO"]
    bots = []
    for i in range(5):
        bots.append({
            "id": i + 2,
            "nombre": nombres[i],
            "modelo": f"Bot-{perfiles[(i + seed_rivales) % 5]}",
            "perfil_bot": perfiles[(i + seed_rivales) % 5],
            "agresividad_seed": rng_b.randint(40, 90),
            "color": colores[i],
            "hp": 100,
            "energia": 100,
            "armaObj": dict(ARMAS["DESARMADO"]),
            "kills": 0,
            "vivo": True,
            "accion": "LISTO",
        })
    return bots


def _decidir_accion_bot(a: Dict[str, Any], entorno: Dict[str, Any], obj_casilla: Any, rng: random.Random) -> str:
    ops = entorno["acciones_validas"]
    if "IR_ZONA_SEGURA" in ops and entorno["niebla"].startswith("FUERA"):
        return "IR_ZONA_SEGURA"
    if "RECOGER_BOTIQUIN" in ops and a["hp"] <= 80:
        return "RECOGER_BOTIQUIN"
    if "EQUIPAR_ARMA" in ops:
        return "EQUIPAR_ARMA"
    if "COMER" in ops and a["energia"] <= 70:
        if obj_casilla and obj_casilla.get("es_veneno") and a.get("agresividad_seed", 60) > 55:
            return "DESCARTAR"
        return "COMER"
    if "ATACAR" in ops:
        if a["hp"] <= 30 and "HUIR" in ops:
            return "HUIR"
        if a.get("perfil_bot") == "DEFENSIVO" and "DEFENDER" in ops and rng.random() < 0.55:
            return "DEFENDER"
        return "ATACAR"
    if "IR_A_OBJETO" in ops:
        return "IR_A_OBJETO"
    if "ACERCARSE_RIVAL" in ops and a["hp"] >= 50:
        return "ACERCARSE_RIVAL"
    if "IR_ZONA_SEGURA" in ops and rng.random() < 0.5:
        return "IR_ZONA_SEGURA"
    return "EXPLORAR"


def _ejecutar_partida_online(agentes: List[Dict[str, Any]], configs_por_id: Dict[int, Dict[str, Any]], seed_mapa: int = 101) -> Dict[str, Any]:
    rng = random.SystemRandom()
    max_turnos = max(10, int(PROFESOR_CONFIG["max_turnos_niebla"]))
    inicio_niebla = int(PROFESOR_CONFIG["turno_inicio_niebla"])
    dano_niebla = int(PROFESOR_CONFIG["dano_niebla"])
    usar_gpu_real = os.environ.get("RUN_REAL_GPU", "0") == "1"

    # Zona segura aleatoria en el mapa 12x12 (no siempre el centro)
    zona_x, zona_y = rng.randint(2, 9), rng.randint(2, 9)
    objetos_mapa = generar_mapa_objetos(rng)
    # Garantizar Espada y Botiquín cerca del punto final de la Zona Segura
    objetos_mapa[(zona_x, zona_y)] = {"tipo": "ARMA", "arma": dict(ARMAS["ESPADA"]), "desc": "Espada (30 daño)"}
    objetos_mapa[(max(0, zona_x - 1), zona_y)] = {"tipo": "BOTIQUIN", "desc": "Botiquín médico (+30 Vida)"}

    # Spawns aleatorios en las 144 casillas, dentro de la zona segura inicial y separados entre sí (dist >= 3)
    radio_seguro = 6
    candidatas = [
        (x, y) for x in range(12) for y in range(12)
        if max(abs(x - zona_x), abs(y - zona_y)) <= radio_seguro
    ]
    rng.shuffle(candidatas)
    elegidas: List[Tuple[int, int]] = []
    for pos in candidatas:
        if all(max(abs(pos[0] - p[0]), abs(pos[1] - p[1])) >= 3 for p in elegidas):
            elegidas.append(pos)
            if len(elegidas) == len(agentes):
                break
    # Si faltan casillas por densidad, completar con las restantes dentro de zona segura
    for pos in candidatas:
        if len(elegidas) == len(agentes):
            break
        if pos not in elegidas:
            elegidas.append(pos)
    for idx, pos in enumerate(elegidas):
        agentes[idx]["x"], agentes[idx]["y"] = pos

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

    try:
        for turno in range(1, max_turnos + 1):
            vivos_inicio = [a for a in agentes if a["vivo"]]
            if len(vivos_inicio) <= 1:
                break
            # Guardar estado previo por si hay desempate en muerte simultánea
            snapshot_previo = {a["id"]: (a["hp"], a["kills"], a["energia"]) for a in vivos_inicio}
            eventos = []

            if turno >= inicio_niebla:
                paso = max(1, (max_turnos - inicio_niebla) // 6)
                if (turno - inicio_niebla) % paso == 0:
                    radio_seguro = max(0 if turno < max_turnos - 1 else -1, radio_seguro - 1)
                    eventos.append(f"⚠️ La Niebla se cierra hacia ({zona_x},{zona_y}) con radio {max(0, radio_seguro)}.")
            if turno == max_turnos:
                radio_seguro = -1
                eventos.append("☠️ ¡LÍMITE DE TURNOS! La Niebla cubre todo el mapa.")

            # 1. Construir el JSON de entorno (Visión = 2 casillas) y decidir acción de cada vivo
            decisiones = {}
            objetivo_cercano_por_id = {}
            recurso_cercano_por_id = {}

            orden_agentes = [a for a in agentes if a["vivo"]]
            rng.shuffle(orden_agentes)

            for a in orden_agentes:
                pos_a = (a["x"], a["y"])
                dist_z = max(abs(a["x"] - zona_x), abs(a["y"] - zona_y))
                fuera_niebla = dist_z > radio_seguro
                dir_zona = _direccion_cardinal(a["x"], a["y"], zona_x, zona_y)

                obj_aqui = objetos_mapa.get(pos_a)
                texto_aqui = obj_aqui["desc"] if obj_aqui else "Nada"

                # Escanear Visión = 2 casillas
                rivales_vista = []
                rival_mas_cerca = None
                min_d_riv = 99
                for otro in agentes:
                    if not otro["vivo"] or otro["id"] == a["id"]:
                        continue
                    d = max(abs(a["x"] - otro["x"]), abs(a["y"] - otro["y"]))
                    if d <= 2:
                        tag_rango = "EN RANGO DE DAÑO (dist=1)" if d <= 1 else "a dist=2"
                        rivales_vista.append(f"{otro['nombre']} {tag_rango} con {otro['armaObj']['nombre']}({otro['armaObj']['dano']}) y {otro['hp']} HP")
                        if d < min_d_riv:
                            min_d_riv = d
                            rival_mas_cerca = otro
                objetivo_cercano_por_id[a["id"]] = (rival_mas_cerca, min_d_riv)

                objetos_vista = []
                mejor_obj_pos = None
                min_d_obj = 99
                for (ox, oy), oinfo in objetos_mapa.items():
                    if (ox, oy) == pos_a:
                        continue
                    d = max(abs(a["x"] - ox), abs(a["y"] - oy))
                    if d <= 2:
                        objetos_vista.append(f"{oinfo['desc']} al {_direccion_cardinal(a['x'], a['y'], ox, oy)} (dist={d})")
                        if d < min_d_obj:
                            min_d_obj = d
                            mejor_obj_pos = (ox, oy)
                recurso_cercano_por_id[a["id"]] = mejor_obj_pos

                # Filtrar acciones válidas según el contexto real
                acciones_validas = []
                if obj_aqui:
                    if obj_aqui["tipo"] == "COMIDA":
                        acciones_validas.extend(["COMER", "DESCARTAR"])
                    elif obj_aqui["tipo"] == "BOTIQUIN" and a["hp"] < 100:
                        acciones_validas.append("RECOGER_BOTIQUIN")
                    elif obj_aqui["tipo"] == "ARMA" and obj_aqui["arma"]["dano"] > a["armaObj"]["dano"]:
                        acciones_validas.append("EQUIPAR_ARMA")
                if rival_mas_cerca and min_d_riv <= 1:
                    acciones_validas.extend(["ATACAR", "DEFENDER"])
                    if a["energia"] >= 20:
                        acciones_validas.append("HUIR")
                if rival_mas_cerca and min_d_riv == 2:
                    acciones_validas.append("ACERCARSE_RIVAL")
                    if a["energia"] >= 20:
                        acciones_validas.append("HUIR")
                if mejor_obj_pos:
                    acciones_validas.append("IR_A_OBJETO")
                if fuera_niebla or dist_z > max(1, radio_seguro - 1):
                    acciones_validas.append("IR_ZONA_SEGURA")
                acciones_validas.append("EXPLORAR")

                entorno_json = {
                    "turno": turno,
                    "vida": a["hp"],
                    "energia": a["energia"],
                    "arma": f"{a['armaObj']['nombre']} ({a['armaObj']['dano']} daño)",
                    "niebla": f"{'PELIGRO FUERA DE ZONA (-20 HP)' if fuera_niebla else 'SEGURA'} (Zona segura hacia {dir_zona})",
                    "en_tu_casilla": texto_aqui,
                    "vista_2_casillas": rivales_vista + objetos_vista,
                    "acciones_validas": acciones_validas,
                }
                a["json_entorno"] = entorno_json

                if a["id"] in modelos_activos:
                    mdl, tok, _, _, _ = modelos_activos[a["id"]]
                    act = inferir_accion_json(mdl, tok, configs_por_id[a["id"]], entorno_json)
                else:
                    act = _decidir_accion_bot(a, entorno_json, obj_aqui, rng)
                decisiones[a["id"]] = act
                a["accion"] = act

            # 2. Ejecutar desgaste de Energía (-10 normal, -20 si HUIR), objetos y movimientos (excepto combate dist<=1)
            atacantes_por_defensor: Dict[int, List[Dict[str, Any]]] = {a["id"]: [] for a in agentes}

            for a in orden_agentes:
                if not a["vivo"]:
                    continue
                act = decisiones[a["id"]]
                gasto_en = 20 if act == "HUIR" else 10
                a["energia"] = max(0, a["energia"] - gasto_en)
                pos_a = (a["x"], a["y"])
                obj_aqui = objetos_mapa.get(pos_a)

                if act == "COMER" and obj_aqui and obj_aqui["tipo"] == "COMIDA":
                    del objetos_mapa[pos_a]
                    if obj_aqui.get("es_veneno"):
                        a["hp"] = max(0, a["hp"] - 20)
                        eventos.append(f"🍄 {a['nombre']} eligió COMER hongos venenosos (-20 HP).")
                    else:
                        a["energia"] = min(100, a["energia"] + 40)
                        eventos.append(f"🍎 {a['nombre']} eligió COMER fruta segura (+40 Energía).")
                elif act == "DESCARTAR" and obj_aqui and obj_aqui["tipo"] == "COMIDA":
                    del objetos_mapa[pos_a]
                    eventos.append(f"🗑️ {a['nombre']} descartó comida sospechosa en su casilla.")
                elif act == "RECOGER_BOTIQUIN" and obj_aqui and obj_aqui["tipo"] == "BOTIQUIN":
                    del objetos_mapa[pos_a]
                    a["hp"] = min(100, a["hp"] + 30)
                    eventos.append(f"💊 {a['nombre']} usó un Botiquín (+30 HP).")
                elif act == "EQUIPAR_ARMA" and obj_aqui and obj_aqui["tipo"] == "ARMA":
                    nueva = dict(obj_aqui["arma"])
                    del objetos_mapa[pos_a]
                    a["armaObj"] = nueva
                    eventos.append(f"{nueva['icono']} {a['nombre']} equipó {nueva['nombre']} ({nueva['dano']} daño).")
                elif act == "HUIR":
                    riv, _ = objetivo_cercano_por_id.get(a["id"], (None, 99))
                    if riv:
                        dx = -1 if riv["x"] > a["x"] else (1 if riv["x"] < a["x"] else rng.choice([-1, 1]))
                        dy = -1 if riv["y"] > a["y"] else (1 if riv["y"] < a["y"] else rng.choice([-1, 1]))
                        a["x"] = max(0, min(11, a["x"] + dx * 2))
                        a["y"] = max(0, min(11, a["y"] + dy * 2))
                    eventos.append(f"💨 {a['nombre']} huyó rápidamente (-20 Energía).")
                elif act == "IR_ZONA_SEGURA":
                    a["x"], a["y"] = _paso_hacia(a["x"], a["y"], zona_x, zona_y)
                    eventos.append(f"🏃 {a['nombre']} avanzó hacia la Zona Segura ({zona_x},{zona_y}).")
                elif act == "IR_A_OBJETO" and recurso_cercano_por_id.get(a["id"]):
                    ox, oy = recurso_cercano_por_id[a["id"]]
                    a["x"], a["y"] = _paso_hacia(a["x"], a["y"], ox, oy)
                    eventos.append(f"🔍 {a['nombre']} fue hacia un objeto cercano.")
                elif act == "ACERCARSE_RIVAL":
                    riv, _ = objetivo_cercano_por_id.get(a["id"], (None, 99))
                    if riv:
                        a["x"], a["y"] = _paso_hacia(a["x"], a["y"], riv["x"], riv["y"])
                        eventos.append(f"⚔️ {a['nombre']} se acercó a {riv['nombre']}.")
                elif act == "ATACAR":
                    riv, d_riv = objetivo_cercano_por_id.get(a["id"], (None, 99))
                    if riv and riv["vivo"] and d_riv <= 1:
                        atacantes_por_defensor[riv["id"]].append(a)
                elif act == "EXPLORAR":
                    # Si ya está fuera de la niebla o al borde, tiende hacia zona segura
                    if max(abs(a["x"] - zona_x), abs(a["y"] - zona_y)) >= radio_seguro:
                        a["x"], a["y"] = _paso_hacia(a["x"], a["y"], zona_x, zona_y)
                    else:
                        a["x"] = max(0, min(11, a["x"] + rng.choice([-1, 0, 1])))
                        a["y"] = max(0, min(11, a["y"] + rng.choice([-1, 0, 1])))

            # 3. Resolver Combates Simultáneos (Regla: DEFENDER bloquea 100% el ataque más fuerte y contraataca 10 HP; el 2do atacante entra completo)
            agentes_por_id = {a["id"]: a for a in agentes}
            for def_id, lista_atq in atacantes_por_defensor.items():
                if not lista_atq:
                    continue
                defensor = agentes_por_id[def_id]
                if not defensor["vivo"]:
                    continue
                # Si el defensor eligió HUIR y ya se alejó a dist > 1, esquiva
                atq_en_rango = [at for at in lista_atq if max(abs(at["x"] - defensor["x"]), abs(at["y"] - defensor["y"])) <= 1]
                if not atq_en_rango:
                    continue

                # Ordenar atacantes por MAYOR daño de arma (y menor vida en caso de empate)
                atq_en_rango.sort(key=lambda x: (-x["armaObj"]["dano"], x["hp"]))

                if decisiones.get(def_id) == "DEFENDER":
                    principal = atq_en_rango[0]
                    principal["hp"] = max(0, principal["hp"] - 10)
                    eventos.append(
                        f"🛡️ {defensor['nombre']} se defendió y bloqueó el 100% de {principal['armaObj']['nombre']} ({principal['armaObj']['dano']}) de {principal['nombre']}, contraatacando (-10 HP a {principal['nombre']})."
                    )
                    if principal["hp"] == 0 and principal["vivo"]:
                        principal["vivo"] = False
                        principal["accion"] = "CAIDO"
                        defensor["kills"] += 1
                        eventos.append(f"💀 {defensor['nombre']} eliminó a {principal['nombre']} con su contraataque.")

                    # Los demás atacantes (si eran 2 o más) entran completos por el flanco
                    for extra in atq_en_rango[1:]:
                        if not defensor["vivo"]:
                            break
                        dmg = extra["armaObj"]["dano"]
                        defensor["hp"] = max(0, defensor["hp"] - dmg)
                        eventos.append(
                            f"⚔️ {extra['nombre']} atacó por el flanco a {defensor['nombre']} mientras bloqueaba a otro (-{dmg} HP con {extra['armaObj']['nombre']})."
                        )
                        if defensor["hp"] == 0:
                            defensor["vivo"] = False
                            defensor["accion"] = "CAIDO"
                            extra["kills"] += 1
                            eventos.append(f"💀 {extra['nombre']} eliminó a {defensor['nombre']}.")
                else:
                    for at in atq_en_rango:
                        if not defensor["vivo"]:
                            break
                        dmg = at["armaObj"]["dano"]
                        defensor["hp"] = max(0, defensor["hp"] - dmg)
                        eventos.append(f"{at['armaObj']['icono']} {at['nombre']} atacó con {at['armaObj']['nombre']} a {defensor['nombre']} (-{dmg} HP).")
                        if defensor["hp"] == 0:
                            defensor["vivo"] = False
                            defensor["accion"] = "CAIDO"
                            at["kills"] += 1
                            eventos.append(f"💀 {at['nombre']} eliminó a {defensor['nombre']}.")

            # 4. Aplicar daño por Energía=0 (-10 HP) y daño por Niebla (-20 HP)
            for a in orden_agentes:
                if not a["vivo"]:
                    continue
                if a["energia"] == 0:
                    a["hp"] = max(0, a["hp"] - 10)
                    eventos.append(f"⚡ {a['nombre']} perdió -10 HP por agotamiento (0 Energía).")
                dist_z = max(abs(a["x"] - zona_x), abs(a["y"] - zona_y))
                if dist_z > radio_seguro:
                    a["hp"] = max(0, a["hp"] - dano_niebla)
                    eventos.append(f"🟪 {a['nombre']} sufrió -{dano_niebla} HP fuera de la Zona Segura.")
                if a["hp"] == 0:
                    a["vivo"] = False
                    a["accion"] = "CAIDO"
                    eventos.append(f"💀 {a['nombre']} cayó en el capítulo {turno}.")

            restantes = [a for a in agentes if a["vivo"]]
            # Desempate garantizado si los últimos murieron al mismo tiempo
            if len(restantes) == 0 and len(vivos_inicio) > 0:
                ganador_desempate = max(vivos_inicio, key=lambda x: snapshot_previo[x["id"]])
                ganador_desempate["vivo"] = True
                ganador_desempate["hp"] = 10
                ganador_desempate["accion"] = "SOBREVIVE_DESEMPATE"
                restantes = [ganador_desempate]
                eventos.append(f"⚖️ ¡Desempate final! {ganador_desempate['nombre']} sobrevive con 10 HP por tener mejor estado previo.")

            if len(restantes) == 1:
                eventos.append(f"🏆 ¡VICTORIA FINAL! {restantes[0]['nombre']} es el único sobreviviente.")

            turnos_log.append({
                "turno": turno,
                "zonaSegura": {"x": zona_x, "y": zona_y, "radio": max(0, radio_seguro)},
                "objetos": [{"x": k[0], "y": k[1], "tipo": v["tipo"], "es_veneno": v.get("es_veneno", False), "desc": v["desc"]} for k, v in objetos_mapa.items()],
                "eventos": eventos if eventos else ["Exploración en la Arena..."],
                "agentes": [{**a, "armaObj": dict(a["armaObj"])} for a in agentes],
            })
            if len(restantes) <= 1:
                break
    finally:
        for m_inst, tok, base_m, mid, is_peft in modelos_activos.values():
            if is_peft and m_inst is not base_m:
                MODELOS_EN_VRAM[mid] = (tok, m_inst.unload())
                MODELOS_EN_VRAM[mid][1].eval()

    ranking = sorted(agentes, key=lambda x: (x["vivo"], x["hp"], x["kills"], x["energia"]), reverse=True)
    return {"turnos_totales": len(turnos_log), "ganador": ranking[0]["nombre"], "ranking": ranking, "turnos": turnos_log}


class EntrenarRequest(BaseModel):
    config: Dict[str, Any]
    reporte_local: Dict[str, Any]
    seed_rivales: int = 101


class ProfesorConfigReq(BaseModel):
    max_turnos_niebla: int = 30
    turno_inicio_niebla: int = 5
    dano_niebla: int = 20
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
    if os.environ.get("RUN_REAL_GPU", "0") == "1":
        with gpu_lock:
            try:
                modo_gpu = ejecutar_entrenamiento_real_gpu(x_team_token, modelo_id, cfg)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error GPU ({type(e).__name__}): {e}")

    ULTIMO_ENTRENAMIENTO_TS[x_team_token] = time.time()
    registro = {
        "token": x_team_token,
        "equipo": cfg.get("equipo", TOKENS_VALIDOS[x_team_token]),
        "modelo_base": modelo_id,
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

    nuestro = {
        "id": 1,
        "nombre": cfg.get("equipo", "Participante_01"),
        "modelo": modelo_id.split("/")[-1],
        "color": "#3B82F6",
        "hp": 100,
        "energia": 100,
        "armaObj": dict(ARMAS["DESARMADO"]),
        "kills": 0,
        "vivo": True,
        "accion": "LISTO",
    }
    bots = generar_bots_con_semilla(int(req.seed_rivales))
    with gpu_lock:
        return _ejecutar_partida_online([nuestro] + bots, {1: cfg_con_token}, int(req.seed_rivales))


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
            "color": colores[i], "hp": 100, "energia": 100,
            "armaObj": dict(ARMAS["DESARMADO"]), "kills": 0, "vivo": True, "accion": "LISTO",
        })
        if "config_completa" in t:
            configs_reales[aid] = {**t["config_completa"], "token": t["token"]}

    bots_extra = generar_bots_con_semilla(seed_rivales)
    while len(agentes) < 12:
        idx = len(agentes)
        b = dict(bots_extra[idx % len(bots_extra)])
        b["id"] = idx + 1
        b["nombre"] = f"Bot_{idx+1:02d}"
        b["color"] = colores[idx]
        agentes.append(b)

    with gpu_lock:
        ULTIMO_REPLAY_PVP = _ejecutar_partida_online(agentes, configs_reales, seed_rivales)
    return ULTIMO_REPLAY_PVP
