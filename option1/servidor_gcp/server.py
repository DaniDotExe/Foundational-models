"""
Servidor GPU (Colab Pro o GCP) para el Reto FM Grand Prix.
Carga UNA sola vez Qwen/Qwen2.5-0.5B en VRAM y ejecuta REALMENTE:
  - Vía 2: ICL (Zero-shot / One-shot / Few-shot + Chain-of-Thought)
  - Vía 3: LoRA (PEFT sobre los ejemplos curados del estudiante)
  - Vía 1: Full Fine-Tuning (sobre copia temporal de W0, dejando W0 intacto)
"""
import copy
import json
import os
import re
import threading
import time
from typing import Any, Dict, List

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_ID = os.environ.get("FM_MODEL_ID", "Qwen/Qwen2.5-0.5B")
MAX_CREDITOS = 100.0
TOKENS_VALIDOS = {f"token_estudiante_{i:02d}": f"Equipo_{i:02d}" for i in range(1, 20)}

app = FastAPI(title="FM Grand Prix - Servidor GPU Real (Qwen2.5-0.5B)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

gpu_lock = threading.Lock()
LEADERBOARD: Dict[str, Dict[str, Any]] = {}
MODELO_CARGADO = {"tokenizer": None, "base_model": None}

# Mini-evaluación real (8 casos por fase para correr rápido en clase)
EVAL_SETS = {
    1: [
        {"input": "Unidad #91 detecta peatón cruzando la calzada en acceso este (dist=18m)", "esperado": "PEATON"},
        {"input": "Unidad #92 reporta barrera caída bloqueando el paso en curva 2 (dist=22m)", "esperado": "OBSTACULO"},
        {"input": "Unidad #93 observa neblina espesa reduciendo visibilidad en túnel (dist=40m)", "esperado": "CLIMA"},
        {"input": "Unidad #94 confirma vía completamente despejada y seca en recta sur (dist=80m)", "esperado": "LIBRE"},
        {"input": "Unidad #95 alerta sobre ciclista sobre el borde del carril en puente (dist=15m)", "esperado": "PEATON"},
        {"input": "Unidad #96 registra roca desprendida en mitad del asfalto en horquilla 5", "esperado": "OBSTACULO"},
    ],
    2: [
        {"input": "En recta principal, vas a V=20 m/s y ves un camión detenido a 38 m. Reacción=0.5*V y Frenado=(V*V)/10. ¿Frenas a tiempo? SI o NO.", "esperado": "NO"},
        {"input": "En asfalto seco, vas a V=10 m/s y ves un cono vial a 22 m. Reacción=0.5*V y Frenado=(V*V)/10. ¿Frenas a tiempo? SI o NO.", "esperado": "SI"},
        {"input": "En bajada norte, vas a V=30 m/s y ves una barrera caída a 90 m. Reacción=0.5*V y Frenado=(V*V)/10. ¿Frenas a tiempo? SI o NO.", "esperado": "NO"},
        {"input": "En curva rápida, vas a V=14 m/s y ves una roca grande a 35 m. Reacción=0.5*V y Frenado=(V*V)/10. ¿Frenas a tiempo? SI o NO.", "esperado": "SI"},
        {"input": "En puente sur, vas a V=24 m/s y ves un auto lento a 60 m. Reacción=0.5*V y Frenado=(V*V)/10. ¿Frenas a tiempo? SI o NO.", "esperado": "NO"},
        {"input": "En tramo nocturno, vas a V=12 m/s y ves un cono vial a 25 m. Reacción=0.5*V y Frenado=(V*V)/10. ¿Frenas a tiempo? SI o NO.", "esperado": "SI"},
    ],
    3: [
        {"input": "zona=final_1 | vel=55 | izq=4.8 | frente=1.1 | der=0.7 | meta=-22deg", "esperado": "ACELERAR_IZQ"},
        {"input": "zona=final_2 | vel=52 | izq=0.8 | frente=1.2 | der=4.6 | meta=+24deg", "esperado": "ACELERAR_DER"},
        {"input": "zona=final_3 | vel=46 | izq=3.2 | frente=6.8 | der=3.3 | meta=0deg", "esperado": "ACELERAR_RECTO"},
        {"input": "zona=final_4 | vel=78 | izq=0.9 | frente=0.8 | der=0.9 | meta=+2deg", "esperado": "FRENAR"},
        {"input": "zona=final_5 | vel=60 | izq=2.2 | frente=2.1 | der=2.2 | meta=-6deg", "esperado": "RODAR_RECTO"},
        {"input": "zona=final_6 | vel=58 | izq=4.5 | frente=1.0 | der=0.8 | meta=-20deg", "esperado": "ACELERAR_IZQ"},
    ],
}


def cargar_modelo_base():
    if MODELO_CARGADO["base_model"] is not None:
        return MODELO_CARGADO["tokenizer"], MODELO_CARGADO["base_model"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[GPU] Cargando {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map="auto")
    mdl.eval()
    MODELO_CARGADO["tokenizer"] = tok
    MODELO_CARGADO["base_model"] = mdl
    return tok, mdl


def _generar_texto(tok, mdl, prompt: str, max_new_tokens: int, temperature: float) -> str:
    import torch
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    do_sample = temperature > 0.05
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 0.01) if do_sample else None,
            pad_token_id=tok.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def _evaluar_modelo(tok, mdl, fase: int, ejemplos_prompt: List[dict], cfg_f: dict) -> dict:
    usar_cot = bool(cfg_f.get("usar_cot", False)) and cfg_f.get("via", "ICL") == "ICL"
    max_tok = int(cfg_f.get("max_tokens", 48))
    temp = float(cfg_f.get("temperature", 0.0))

    prefijo = ""
    for ej in ejemplos_prompt:
        if usar_cot and ej.get("rationale"):
            prefijo += f"Entrada: {ej['input']}\nRazonamiento: {ej['rationale']}\nSalida: {ej['output']}\n\n"
        else:
            prefijo += f"Entrada: {ej['input']}\nSalida: {ej['output']}\n\n"

    aciertos = 0
    muestras = []
    for item in EVAL_SETS[fase]:
        prompt = prefijo + f"Entrada: {item['input']}\n" + ("Razonamiento:" if usar_cot else "Salida:")
        pred = _generar_texto(tok, mdl, prompt, max_new_tokens=max_tok, temperature=temp)
        pred_line = pred.split("\n\n")[0].strip()
        ok = False
        if fase == 1:
            ok = item["esperado"] in pred_line.upper() and "{" in pred_line and "}" in pred_line
        elif fase == 2:
            m = re.findall(r"\b(SI|NO)\b", pred_line.upper())
            ok = bool(m) and m[-1] == item["esperado"]
        else:
            ok = item["esperado"] in pred_line.upper()
        if ok:
            aciertos += 1
        if len(muestras) < 2:
            muestras.append({"esperado": item["esperado"], "prediccion": pred_line[:140], "ok": ok})

    score = round(100.0 * aciertos / len(EVAL_SETS[fase]), 1)
    return {"score": score, "aciertos": aciertos, "total": len(EVAL_SETS[fase]), "muestras": muestras}


def _entrenar_sft(tok, mdl, ejemplos: List[dict], epochs: int, lr: float):
    import torch
    mdl.train()
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad], lr=lr)
    subset = ejemplos[:40]
    for _ in range(max(1, min(epochs, 6))):
        for ej in subset:
            txt = f"Entrada: {ej['input']}\nSalida: {ej['output']}{tok.eos_token}"
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=256).to(mdl.device)
            out = mdl(**enc, labels=enc["input_ids"])
            out.loss.backward()
            opt.step()
            opt.zero_grad()
    mdl.eval()


def _correr_fase_gpu(tok, base_mdl, fase: int, ejemplos: List[dict], cfg_f: dict) -> dict:
    import torch
    via = cfg_f.get("via", "ICL").upper()
    if via == "ICL":
        modo = cfg_f.get("modo_icl", "zero_shot")
        k = 0 if modo == "zero_shot" else (1 if modo == "one_shot" else int(cfg_f.get("k_ejemplos", 4)))
        return _evaluar_modelo(tok, base_mdl, fase, ejemplos[:k], cfg_f)
    elif via == "LORA":
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=int(cfg_f.get("lora_r", 8)),
            lora_alpha=int(cfg_f.get("lora_alpha", 16)),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_mdl = get_peft_model(base_mdl, lcfg)
        _entrenar_sft(tok, peft_mdl, ejemplos, int(cfg_f.get("epochs", 3)), float(cfg_f.get("lr", 5e-4)))
        res = _evaluar_modelo(tok, peft_mdl, fase, [], cfg_f)
        # Limpiar adaptador para dejar W0 intacto
        base_mdl.delete_adapter("default") if hasattr(base_mdl, "delete_adapter") else None
        del peft_mdl
        torch.cuda.empty_cache()
        return res
    else:  # FULL_FT
        mdl_copia = copy.deepcopy(base_mdl)
        _entrenar_sft(tok, mdl_copia, ejemplos, int(cfg_f.get("epochs", 2)), float(cfg_f.get("lr", 5e-5)))
        res = _evaluar_modelo(tok, mdl_copia, fase, [], cfg_f)
        del mdl_copia
        torch.cuda.empty_cache()
        return res


class MisionRequest(BaseModel):
    config: Dict[str, Any]
    resumen_local: Dict[str, Any] = {}
    ejemplos_curados: Dict[str, List[dict]] = {}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "real_gpu": os.environ.get("RUN_REAL_GPU", "0") == "1", "equipos": len(LEADERBOARD)}


@app.get("/leaderboard")
def get_leaderboard():
    return {"leaderboard": sorted(LEADERBOARD.values(), key=lambda x: x["puntaje_total"], reverse=True)}


@app.post("/ejecutar_mision")
def ejecutar_mision(req: MisionRequest, x_team_token: str = Header(default="")):
    if x_team_token not in TOKENS_VALIDOS:
        raise HTTPException(status_code=401, detail="TEAM_TOKEN inválido o no autorizado.")

    cfg = req.config
    equipo = cfg.get("equipo", TOKENS_VALIDOS[x_team_token])
    resumen = req.resumen_local
    creditos = sum(resumen.get(f"fase{i}", {}).get("costo", {}).get("creditos", 0.0) for i in (1, 2, 3))
    if creditos > MAX_CREDITOS:
        raise HTTPException(status_code=400, detail=f"Presupuesto excedido: {creditos:.1f} / {MAX_CREDITOS} créditos.")

    t0 = time.time()
    if os.environ.get("RUN_REAL_GPU", "0") == "1":
        with gpu_lock:
            tok, base_mdl = cargar_modelo_base()
            r1 = _correr_fase_gpu(tok, base_mdl, 1, req.ejemplos_curados.get("fase1", []), cfg.get("fase1", {}))
            r2 = _correr_fase_gpu(tok, base_mdl, 2, req.ejemplos_curados.get("fase2", []), cfg.get("fase2", {}))
            r3 = _correr_fase_gpu(tok, base_mdl, 3, req.ejemplos_curados.get("fase3", []), cfg.get("fase3", {}))
            s1, s2, s3 = r1["score"], r2["score"], r3["score"]
            muestras = {"fase1": r1["muestras"], "fase2": r2["muestras"], "fase3": r3["muestras"]}
    else:
        s1, s2, s3 = 85.0, 85.0, 85.0
        muestras = {"aviso": "Ejecutado sin RUN_REAL_GPU=1 (modo demo)"}

    bono = round(max(0.0, (MAX_CREDITOS - creditos) * 0.4), 1) if (s1 >= 75 and s2 >= 75 and s3 >= 75) else 0.0
    total = round((s1 + s2 + s3) / 3.0 + bono, 1)

    registro = {
        "equipo": equipo,
        "fase1_score": s1,
        "fase2_score": s2,
        "fase3_score": s3,
        "creditos_usados": round(creditos, 1),
        "bono_eficiencia": bono,
        "puntaje_total": total,
        "tiempo_gpu_s": round(time.time() - t0, 2),
        "muestras": muestras,
    }
    LEADERBOARD[x_team_token] = registro
    return registro
