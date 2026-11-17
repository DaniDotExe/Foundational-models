"""
Servidor FastAPI para Google Colab Pro (1x NVIDIA L4 24GB o A100 40GB).
Soporta los 3 modelos base (Qwen2.5-0.5B / 1.5B / 3B-Instruct), valida el límite
de 1.0 PFLOP (100 créditos) para los 12 equipos y ejecuta la Arena PvP Final de 25 turnos.
"""
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODELOS_VALIDOS = {
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
}
MAX_CREDITOS = 100.0
TOKENS_VALIDOS = {f"token_estudiante_{i:02d}": f"Distrito_{i:02d}" for i in range(1, 13)}

SAVE_DIR = Path(os.environ.get("ARENA_SAVE_DIR", "./estado_arena"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Juegos del Hambre FM - Servidor Colab Pro (0.5B / 1.5B / 3B)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gpu_lock = threading.Lock()
TRIBUTOS: Dict[str, Dict[str, Any]] = {}
MODELOS_EN_VRAM: Dict[str, Any] = {}
ULTIMO_REPLAY_PVP: Dict[str, Any] = {}


def obtener_modelo_gpu(model_id: str):
    """Carga bajo demanda uno de los 3 modelos en bfloat16 dentro de la L4/A100."""
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


class EntrenarRequest(BaseModel):
    config: Dict[str, Any]
    reporte_local: Dict[str, Any]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tributos_registrados": len(TRIBUTOS),
        "modelos_cargados": list(MODELOS_EN_VRAM.keys()),
    }


@app.get("/leaderboard")
def get_leaderboard():
    lista = sorted(TRIBUTOS.values(), key=lambda x: x["media_radar"] + x["bono_eficiencia"], reverse=True)
    return {"tributos": lista, "ultimo_pvp": ULTIMO_REPLAY_PVP.get("ranking", [])}


@app.post("/entrenar_tributo")
def entrenar_tributo(req: EntrenarRequest, x_team_token: str = Header(default="")):
    if x_team_token not in TOKENS_VALIDOS:
        raise HTTPException(status_code=401, detail="TEAM_TOKEN inválido (usa token_estudiante_01 .. 12).")

    cfg = req.config
    rep = req.reporte_local
    modelo_id = cfg.get("modelo_base", "Qwen/Qwen2.5-1.5B-Instruct")
    if modelo_id not in MODELOS_VALIDOS:
        raise HTTPException(status_code=400, detail=f"Modelo no permitido: {modelo_id}")

    creditos = float(rep.get("creditos_totales", 999.0))
    if creditos > MAX_CREDITOS or not rep.get("valido", False):
        raise HTTPException(status_code=400, detail=f"Configuración inválida o excede {MAX_CREDITOS} créditos.")

    t0 = time.time()
    if os.environ.get("RUN_REAL_GPU", "0") == "1":
        with gpu_lock:
            tok, mdl = obtener_modelo_gpu(modelo_id)
            _ = (tok, mdl)

    radar = rep.get("radar_stats", {"comer": 60, "guerra": 60, "correr": 60, "sigilo": 60})
    bono = round(max(0.0, (MAX_CREDITOS - creditos) * 0.25), 1)
    media = round(sum(radar.values()) / 4.0, 1)

    registro = {
        "token": x_team_token,
        "equipo": cfg.get("equipo", TOKENS_VALIDOS[x_team_token]),
        "modelo_base": modelo_id,
        "doctrina": cfg.get("doctrina", "BOSQUE"),
        "creditos_usados": creditos,
        "bono_eficiencia": bono,
        "radar_stats": radar,
        "media_radar": media,
        "vias": {k: cfg.get("stats", {}).get(k, {}).get("via", "ICL") for k in ("comer", "guerra", "correr", "sigilo")},
        "tiempo_gpu_s": round(time.time() - t0, 2),
    }
    TRIBUTOS[x_team_token] = registro
    with open(SAVE_DIR / f"{x_team_token}.json", "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)
    return {"status": "ok", "tributo": registro}


@app.post("/correr_arena_pvp")
def correr_arena_pvp():
    """Ejecuta la Arena Final PvP de 12 tributos en cuadrícula 12x12 durante 25 turnos."""
    global ULTIMO_REPLAY_PVP
    rng = random.Random(2026)

    spawns = [
        (5, 4), (6, 4), (7, 5), (7, 6), (6, 7), (5, 7),
        (4, 6), (4, 5), (2, 2), (9, 2), (2, 9), (9, 9),
    ]
    colores = [
        "#F43F5E", "#14B8A6", "#F59E0B", "#818CF8", "#10B981", "#EC4899",
        "#38BDF8", "#A855F7", "#84CC16", "#FB923C", "#2DD4BF", "#E879F9",
    ]

    lista_tributos: List[Dict[str, Any]] = list(TRIBUTOS.values())
    # Si hay menos de 12 estudiantes conectados, rellenar con Distritos Bot para completar los 12
    for idx in range(len(lista_tributos), 12):
        d_num = idx + 1
        m_bot = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"][idx % 3]
        doc_bot = ["CORNUCOPIA", "BOSQUE", "CAZADOR"][idx % 3]
        lista_tributos.append({
            "token": f"bot_{d_num:02d}",
            "equipo": f"Distrito_{d_num:02d}_Bot",
            "modelo_base": m_bot,
            "doctrina": doc_bot,
            "creditos_usados": 88.0,
            "bono_eficiencia": 3.0,
            "radar_stats": {
                "comer": 62 + (idx * 5) % 28,
                "guerra": 60 + (idx * 7) % 30,
                "correr": 64 + (idx * 3) % 26,
                "sigilo": 58 + (idx * 9) % 30,
            },
        })

    agentes = []
    for i, t in enumerate(lista_tributos[:12]):
        sx, sy = spawns[i]
        if t["doctrina"] == "BOSQUE":
            sx = 1 if sx < 6 else 10
            sy = 1 if sy < 6 else 10
        bono = int(t.get("bono_eficiencia", 0))
        agentes.append({
            "id": i + 1,
            "nombre": t["equipo"],
            "modelo": t["modelo_base"].split("/")[-1],
            "doctrina": t["doctrina"],
            "color": colores[i],
            "x": sx,
            "y": sy,
            "hp": 100,
            "hambre": min(100, 85 + bono),
            "estamina": min(100, 85 + bono),
            "arma_bono": 0,
            "kills": 0,
            "vivo": True,
            "ultima_accion": "LISTO",
            "stats": t["radar_stats"],
        })

    turnos_log = []
    radio_seguro = 6

    for turno in range(1, 26):
        eventos = []
        if turno >= 6 and turno % 4 == 2:
            radio_seguro = max(2, radio_seguro - 1)
            eventos.append(f"⚠️ Niebla del Capitolio se reduce a radio {radio_seguro}.")

        bestias = (turno % 5 == 0)
        if bestias:
            eventos.append("🐺 ¡Bestias Mutantes patrullan el borde del mapa!")

        for a in agentes:
            if not a["vivo"]:
                continue

            a["hambre"] = max(0, a["hambre"] - 6)
            a["estamina"] = min(100, a["estamina"] + 4)

            # Movimiento hacia centro o exploración
            dist_c = max(abs(a["x"] - 5.5), abs(a["y"] - 5.5))
            if turno == 1 and a["doctrina"] == "CORNUCOPIA":
                a["x"], a["y"] = 5 + rng.randint(0, 1), 5 + rng.randint(0, 1)
                if a["stats"]["guerra"] + rng.randint(-10, 12) >= 72:
                    a["arma_bono"] = 18
                    a["ultima_accion"] = "SAQUEAR_CORNUCOPIA"
                    eventos.append(f"⚔️ {a['nombre']} obtiene arma (+18 ATQ) en la Cornucopia.")
                else:
                    a["hp"] -= 18
                    a["ultima_accion"] = "HERIDO_CORNUCOPIA"
                    eventos.append(f"💥 {a['nombre']} recibe -18 HP en el caos de la Cornucopia.")
            elif dist_c > radio_seguro:
                if a["stats"]["correr"] + rng.randint(-10, 10) >= 70 and a["estamina"] >= 10:
                    a["estamina"] -= 10
                    a["x"] += 1 if a["x"] < 5 else (-1 if a["x"] > 6 else 0)
                    a["y"] += 1 if a["y"] < 5 else (-1 if a["y"] > 6 else 0)
                    a["ultima_accion"] = "CORRER_ZONA"
                else:
                    a["hp"] -= 16
                    a["ultima_accion"] = "DANO_NIEBLA"
                    eventos.append(f"☠️ {a['nombre']} sufre -16 HP por la Niebla del Capitolio.")
            else:
                dx, dy = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)])
                a["x"] = max(0, min(11, a["x"] + dx))
                a["y"] = max(0, min(11, a["y"] + dy))
                a["ultima_accion"] = "PATRULLAR"

            # Comer si tiene hambre
            if a["hambre"] < 62 and a["vivo"]:
                if a["stats"]["comer"] + rng.randint(-12, 12) >= 65:
                    a["hambre"] = min(100, a["hambre"] + 28)
                    a["hp"] = min(100, a["hp"] + 5)
                    a["ultima_accion"] = "COMER_BAYAS"
                    eventos.append(f"🫐 {a['nombre']} recolecta alimento seguro (+28 Hambre).")
                else:
                    a["hp"] -= 18
                    a["ultima_accion"] = "VENENO"
                    eventos.append(f"🍄 {a['nombre']} come bayas venenosas por ruido en datos (-18 HP).")

            if bestias and dist_c >= radio_seguro - 0.5:
                if a["stats"]["sigilo"] + rng.randint(-10, 10) < 68:
                    a["hp"] -= 15
                    a["ultima_accion"] = "ATAQUE_BESTIA"
                    eventos.append(f"🐺 Bestia mutante hiere a {a['nombre']} (-15 HP).")

            if a["hambre"] == 0:
                a["hp"] -= 12

            if a["hp"] <= 0:
                a["hp"] = 0
                a["vivo"] = False
                a["ultima_accion"] = "ELIMINADO"
                eventos.append(f"💀 ¡Cañonazo! {a['nombre']} cae en el turno {turno}.")

        # Combates PvP entre tributos cercanos
        vivos = [x for x in agentes if x["vivo"]]
        rng.shuffle(vivos)
        for i in range(0, len(vivos) - 1, 2):
            p1, p2 = vivos[i], vivos[i + 1]
            if abs(p1["x"] - p2["x"]) <= 2 and abs(p1["y"] - p2["y"]) <= 2:
                s1 = p1["stats"]["guerra"] + p1["arma_bono"] + rng.randint(-12, 12)
                s2 = p2["stats"]["guerra"] + p2["arma_bono"] + rng.randint(-12, 12)
                gan, per = (p1, p2) if s1 >= s2 else (p2, p1)
                if per["stats"]["correr"] + per["stats"]["sigilo"] > 158 and per["estamina"] >= 14:
                    per["estamina"] -= 14
                    per["ultima_accion"] = "EVADIR"
                    eventos.append(f"💨 {per['nombre']} evade el ataque de {gan['nombre']}.")
                else:
                    dano = max(14, min(36, int((gan["stats"]["guerra"] + gan["arma_bono"]) * 0.28)))
                    per["hp"] = max(0, per["hp"] - dano)
                    gan["ultima_accion"] = "ATACAR"
                    eventos.append(f"⚔️ {gan['nombre']} golpea a {per['nombre']} (-{dano} HP).")
                    if per["hp"] == 0:
                        per["vivo"] = False
                        per["ultima_accion"] = "ELIMINADO"
                        gan["kills"] += 1
                        eventos.append(f"💀 ¡Cañonazo! {gan['nombre']} elimina a {per['nombre']}.")

        turnos_log.append({
            "turno": turno,
            "radio_seguro": radio_seguro,
            "bestias": bestias,
            "eventos": eventos,
            "agentes": [dict(a) for a in agentes],
        })

        if sum(1 for a in agentes if a["vivo"]) <= 1:
            break

    ranking = sorted(agentes, key=lambda x: (x["vivo"], x["hp"], x["kills"], sum(x["stats"].values())), reverse=True)
    ULTIMO_REPLAY_PVP = {
        "turnos_totales": len(turnos_log),
        "ganador": ranking[0]["nombre"],
        "ranking": [
            {
                "puesto": idx + 1,
                "nombre": r["nombre"],
                "modelo": r["modelo"],
                "doctrina": r["doctrina"],
                "vivo": r["vivo"],
                "hp": r["hp"],
                "kills": r["kills"],
                "stats": r["stats"],
            }
            for idx, r in enumerate(ranking)
        ],
        "turnos": turnos_log,
    }
    with open(SAVE_DIR / "arena_pvp_replay.json", "w", encoding="utf-8") as f:
        json.dump(ULTIMO_REPLAY_PVP, f, indent=2, ensure_ascii=False)
    return ULTIMO_REPLAY_PVP
