"""
Cliente del estudiante (Option 2 · Juegos del Hambre de Modelos Fundacionales):
1. Valida config.py: verifica modelo base (0.5B / 1.5B / 3B), restricción de FULL_FT en 0.5B,
   y calcula FLOPs reales (límite 1.0 PFLOP = 100 créditos).
2. Ejecuta la tubería de curación (MinHash 3-gramas + filtro de ruido) sobre las 4 stats.
3. Calcula el Radar de Stats (Comer, Guerra, Correr, Sigilo) y corre una Simulación Local GRATIS
   de 25 turnos contra 5 Agentes Imaginarios, guardando resultados/ultimo_intento.json.
4. Con --enviar-colab envía el entrenamiento real al servidor en Google Colab Pro.
"""
import argparse
import json
import random
from pathlib import Path
import urllib.request
import urllib.error

from config import CONFIG
import generar_datos

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
RES_DIR = BASE_DIR / "resultados"
RES_DIR.mkdir(exist_ok=True)

MODELOS_SOPORTADOS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {"n_params": 0.494e9, "vram_gb": 1.0, "bono_icl": 0, "bono_cot": 4},
    "Qwen/Qwen2.5-1.5B-Instruct": {"n_params": 1.540e9, "vram_gb": 3.0, "bono_icl": 12, "bono_cot": 16},
    "Qwen/Qwen2.5-3B-Instruct":   {"n_params": 3.090e9, "vram_gb": 6.0, "bono_icl": 22, "bono_cot": 26},
}

MAX_FLOPS = 1.0e15          # 1.0 PetaFLOP = 1,000 TFLOPs = 100 Créditos
MAX_CREDITOS = 100.0
GPU_L4_FLOPS = 4.5e13       # ~45 TFLOP/s efectivos sostenidos en L4

VIAS_PRESET = {
    "ICL":           {"epochs": 0, "lora_r": 0},
    "LORA_LIGERO":   {"epochs": 2, "lora_r": 8},
    "LORA_PROFUNDO": {"epochs": 4, "lora_r": 16},
    "FULL_FT":       {"epochs": 3, "lora_r": 0},
}


def asegurar_datasets():
    archivos = ["stat1_comer.jsonl", "stat2_guerra.jsonl", "stat3_correr.jsonl", "stat4_sigilo.jsonl"]
    if not all((DATOS_DIR / a).exists() for a in archivos):
        generar_datos.generar_stat1_comer()
        generar_datos.generar_stat2_guerra()
        generar_datos.generar_stat3_correr()
        generar_datos.generar_stat4_sigilo()


def _shingles(texto: str, k: int = 5) -> set:
    palabras = texto.lower().split()
    pares = {f"{palabras[i]}_{palabras[i+1]}" for i in range(len(palabras) - 1)}
    return pares | set(palabras)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _es_ejemplo_limpio(stat_key: str, row: dict) -> bool:
    out = str(row.get("output", "")).strip()
    inp = str(row.get("input", "")).lower()
    rat = str(row.get("rationale", "")).lower()

    if stat_key == "comer":
        if not (out.startswith("{") and out.endswith("}")):
            return False
        if ("almendra amarga" in inp or "espuma" in inp or "cable explosivo" in inp) and '"COMER"' in out:
            return False
        return True
    elif stat_key == "guerra":
        if out not in ("ATACAR", "DEFENDER", "EMBOSCAR"):
            return False
        if "hp_propia=15" in inp and out == "ATACAR":
            return False
        return True
    elif stat_key == "correr":
        if out not in ("SPRINT_CENTRO", "TROTE_CENTRO", "REFUGIO_RECUPERAR"):
            return False
        if "cuenta rápida" in rat or ("estamina=20" in inp and out == "SPRINT_CENTRO"):
            return False
        return True
    else:  # sigilo
        if out not in ("OCULTARSE", "SAQUEAR", "ACECHAR"):
            return False
        if "rodeando la caja" in inp and out == "SAQUEAR":
            return False
        return True


def curar_dataset(ruta_jsonl: Path, stat_key: str, cur_cfg: dict) -> dict:
    raw = [json.loads(line) for line in open(ruta_jsonl, encoding="utf-8") if line.strip()]
    sobrevivientes = []
    vistas = []

    for r in raw:
        if cur_cfg.get("filtrar_ruido", True) and not _es_ejemplo_limpio(stat_key, r):
            continue
        if cur_cfg.get("dedup_minhash", True):
            sh = _shingles(r["input"])
            umbral = float(cur_cfg.get("umbral_minhash", 0.82))
            if any(_jaccard(sh, v) >= umbral for v in vistas):
                continue
            vistas.append(sh)
        sobrevivientes.append(r)

    frac = max(0.2, min(1.0, float(cur_cfg.get("fraccion_datos_D", 1.0))))
    n_final = max(1, int(len(sobrevivientes) * frac))
    usados = sobrevivientes[:n_final]
    ruido_restante = sum(1 for x in usados if x.get("meta_tipo") == "ruido")
    return {
        "total_crudo": len(raw),
        "tras_filtros": len(sobrevivientes),
        "usados_D": len(usados),
        "ruido_restante": ruido_restante,
    }


def calcular_flops_stat(modelo_id: str, stat_key: str, cfg_stat: dict, cur_info: dict, cur_cfg: dict) -> dict:
    info_m = MODELOS_SOPORTADOS[modelo_id]
    n_params = info_m["n_params"]
    via = cfg_stat.get("via", "ICL").upper()
    n_datos = cur_info["usados_D"]
    tokens_por_ejemplo = 110
    n_eval_turnos = 25

    if via == "ICL":
        modo = cfg_stat.get("modo_icl", "few_shot")
        k = 0 if modo == "zero_shot" else (1 if modo == "one_shot" else int(cfg_stat.get("k_ejemplos", 4)))
        cot = bool(cfg_stat.get("usar_cot", False))
        tokens_prompt = 60 + k * (tokens_por_ejemplo + (55 if cot else 0))
        tokens_out = 95 if cot else 24
        flops_train = 0.0
        # Costo de contexto e inferencia de preparación/validación
        flops_inf = 2.0 * n_params * n_eval_turnos * (tokens_prompt + tokens_out) * 1.8
    elif via in ("LORA_LIGERO", "LORA_PROFUNDO"):
        preset = VIAS_PRESET[via]
        epochs = preset["epochs"]
        r = preset["lora_r"]
        tokens_D = n_datos * tokens_por_ejemplo * epochs * 3.2
        flops_train = (4.0 + 0.08 * (r / 8.0)) * n_params * tokens_D
        flops_inf = 2.0 * n_params * (n_eval_turnos * 65)
    elif via == "FULL_FT":
        epochs = VIAS_PRESET["FULL_FT"]["epochs"]
        tokens_D = n_datos * tokens_por_ejemplo * epochs * 3.5
        flops_train = 6.0 * n_params * tokens_D
        flops_inf = 2.0 * n_params * (n_eval_turnos * 65)
    else:
        raise ValueError(f"Vía desconocida: {via}")

    flops_totales = flops_train + flops_inf
    creditos = round((flops_totales / MAX_FLOPS) * 100.0, 1)

    # Cálculo del puntaje de la stat (0 - 100) según modelo, vía y curación
    pen_ruido = 24 if not cur_cfg.get("filtrar_ruido", False) else 0
    pen_dedup = 9 if not cur_cfg.get("dedup_minhash", False) else 0
    umbral = float(cur_cfg.get("umbral_minhash", 0.82))
    pen_umbral = 18 if umbral < 0.65 else (6 if umbral > 0.93 else 0)
    bono_datos = min(12, int(n_datos * 0.28))

    if via == "ICL":
        modo = cfg_stat.get("modo_icl", "few_shot")
        k = 0 if modo == "zero_shot" else (1 if modo == "one_shot" else int(cfg_stat.get("k_ejemplos", 4)))
        cot = bool(cfg_stat.get("usar_cot", False))
        base = 42 + info_m["bono_icl"] + min(18, k * 4)
        if cot:
            base += info_m["bono_cot"] if stat_key in ("correr", "sigilo") else (info_m["bono_cot"] // 2)
        elif stat_key == "correr":
            base -= 14  # Correr requiere cálculo aritmético de estamina
        score = base - (pen_ruido // 2) - pen_umbral
    elif via == "LORA_LIGERO":
        base = 74 + bono_datos + (4 if n_params > 1e9 else 0)
        if stat_key == "correr" and not cfg_stat.get("usar_cot", False):
            base -= 6
        score = base - pen_ruido - pen_dedup - pen_umbral
    elif via == "LORA_PROFUNDO":
        base = 84 + bono_datos + (5 if n_params > 1e9 else 0)
        score = base - pen_ruido - pen_dedup - pen_umbral
    else:  # FULL_FT
        base = 88 + bono_datos
        score = base - int(pen_ruido * 1.2) - pen_dedup - pen_umbral

    score = max(15, min(99, int(round(score))))
    return {
        "via": via,
        "flops_totales": flops_totales,
        "tflops": round(flops_totales / 1e12, 2),
        "creditos": creditos,
        "tiempo_l4_s": round(flops_totales / GPU_L4_FLOPS, 2),
        "score_stat": score,
    }


def simular_arena_local(equipo: str, modelo_id: str, stats_scores: dict, doctrina: str, bono_eficiencia: float) -> dict:
    """Simula 25 turnos en un mapa 12x12 contra 5 Agentes Imaginarios (0 créditos)."""
    rng = random.Random(42 + sum(stats_scores.values()))
    participantes = [
        {
            "nombre": equipo,
            "es_jugador": True,
            "modelo": modelo_id.split("/")[-1],
            "doctrina": doctrina,
            "x": 5 if doctrina == "CORNUCOPIA" else 1,
            "y": 5 if doctrina == "CORNUCOPIA" else 1,
            "hp": 100,
            "hambre": min(100, 85 + int(bono_eficiencia)),
            "estamina": min(100, 85 + int(bono_eficiencia)),
            "arma_bono": 0,
            "eliminaciones": 0,
            "vivo": True,
            "stats": stats_scores,
        },
        {"nombre": "Cato_Carnicero", "es_jugador": False, "modelo": "Bot-0.5B-FullFT", "doctrina": "CORNUCOPIA", "x": 6, "y": 5, "hp": 100, "hambre": 80, "estamina": 75, "arma_bono": 0, "eliminaciones": 0, "vivo": True, "stats": {"comer": 55, "guerra": 92, "correr": 60, "sigilo": 50}},
        {"nombre": "Foxface_Astuta", "es_jugador": False, "modelo": "Bot-3B-ICL", "doctrina": "BOSQUE", "x": 10, "y": 2, "hp": 100, "hambre": 85, "estamina": 90, "arma_bono": 0, "eliminaciones": 0, "vivo": True, "stats": {"comer": 90, "guerra": 52, "correr": 88, "sigilo": 91}},
        {"nombre": "Rue_Exploradora", "es_jugador": False, "modelo": "Bot-1.5B-LoRA", "doctrina": "BOSQUE", "x": 2, "y": 9, "hp": 100, "hambre": 85, "estamina": 85, "arma_bono": 0, "eliminaciones": 0, "vivo": True, "stats": {"comer": 86, "guerra": 58, "correr": 84, "sigilo": 88}},
        {"nombre": "Thresh_Tanque", "es_jugador": False, "modelo": "Bot-1.5B-LoRA", "doctrina": "CAZADOR", "x": 9, "y": 9, "hp": 100, "hambre": 80, "estamina": 80, "arma_bono": 0, "eliminaciones": 0, "vivo": True, "stats": {"comer": 72, "guerra": 86, "correr": 70, "sigilo": 68}},
        {"nombre": "Novato_SinCurar", "es_jugador": False, "modelo": "Bot-0.5B-Ruidoso", "doctrina": "CORNUCOPIA", "x": 6, "y": 6, "hp": 100, "hambre": 75, "estamina": 70, "arma_bono": 0, "eliminaciones": 0, "vivo": True, "stats": {"comer": 35, "guerra": 45, "correr": 40, "sigilo": 38}},
    ]

    historial_turnos = []
    radio_seguro = 6  # Centro en (5.5, 5.5)

    for turno in range(1, 26):
        eventos_turno = []
        if turno >= 6 and turno % 4 == 2:
            radio_seguro = max(2, radio_seguro - 1)
            eventos_turno.append(f"⚠️ La Niebla del Capitolio se cierra a radio {radio_seguro}.")

        bestia_activa = (turno % 5 == 0)
        if bestia_activa:
            eventos_turno.append("🐺 ¡El Capitolio suelta Bestias Mutantes en el anillo exterior!")

        # Turno 1: Cornucopia
        if turno == 1:
            for p in participantes:
                if p["vivo"] and p["doctrina"] == "CORNUCOPIA":
                    if p["stats"]["guerra"] + rng.randint(-10, 15) >= 70:
                        p["arma_bono"] = 18
                        p["hambre"] = min(100, p["hambre"] + 15)
                        eventos_turno.append(f"⚔️ {p['nombre']} conquista un arma (+18 ATQ) en la Cornucopia.")
                    else:
                        p["hp"] -= 22
                        eventos_turno.append(f"💥 {p['nombre']} sufre heridas (-22 HP) disputando la Cornucopia.")

        # Acciones individuales
        for p in participantes:
            if not p["vivo"]:
                continue

            p["hambre"] = max(0, p["hambre"] - 6)
            dist_centro = max(abs(p["x"] - 5.5), abs(p["y"] - 5.5))

            # Daño por niebla fuera de zona segura
            if dist_centro > radio_seguro:
                if p["stats"]["correr"] + rng.randint(-10, 10) >= 72 and p["estamina"] >= 12:
                    p["estamina"] -= 12
                    p["x"] += 1 if p["x"] < 5 else (-1 if p["x"] > 6 else 0)
                    p["y"] += 1 if p["y"] < 5 else (-1 if p["y"] > 6 else 0)
                    eventos_turno.append(f"🏃 {p['nombre']} calcula ruta de escape y entra a la zona segura.")
                else:
                    p["hp"] -= 16
                    eventos_turno.append(f"☠️ {p['nombre']} respira niebla tóxica (-16 HP) por mal cálculo de evasión.")

            # Forrajeo si tiene hambre
            if p["hambre"] < 65:
                if p["stats"]["comer"] + rng.randint(-12, 12) >= 65:
                    p["hambre"] = min(100, p["hambre"] + 28)
                    p["hp"] = min(100, p["hp"] + 4)
                    eventos_turno.append(f"🫐 {p['nombre']} identifica bayas seguras (+28 Hambre).")
                else:
                    p["hp"] -= 18
                    eventos_turno.append(f"🍄 {p['nombre']} se intoxica con bayas venenosas (-18 HP).")

            # Evento de bestias
            if bestia_activa and dist_centro >= radio_seguro - 0.5:
                if p["stats"]["sigilo"] + rng.randint(-10, 10) < 68:
                    p["hp"] -= 15
                    eventos_turno.append(f"🐺 Bestia mutante detecta a {p['nombre']} (-15 HP).")

            # Hambre extrema
            if p["hambre"] == 0:
                p["hp"] -= 12
                eventos_turno.append(f"🥀 {p['nombre']} pierde -12 HP por inanición.")

            if p["hp"] <= 0:
                p["hp"] = 0
                p["vivo"] = False
                eventos_turno.append(f"💀 ¡Cañonazo! {p['nombre']} ha caído en el turno {turno}.")

        # Combates entre vivos cercanos
        vivos = [p for p in participantes if p["vivo"]]
        if len(vivos) >= 2 and turno >= 2:
            a, b = rng.sample(vivos, 2)
            poder_a = a["stats"]["guerra"] + a["arma_bono"] + rng.randint(-12, 12)
            poder_b = b["stats"]["guerra"] + b["arma_bono"] + rng.randint(-12, 12)
            if abs(poder_a - poder_b) > 8:
                ganador, perdedor = (a, b) if poder_a > poder_b else (b, a)
                if perdedor["stats"]["correr"] + perdedor["stats"]["sigilo"] > 155 and perdedor["estamina"] >= 15:
                    perdedor["estamina"] -= 15
                    eventos_turno.append(f"💨 {perdedor['nombre']} esquiva una emboscada de {ganador['nombre']}.")
                else:
                    dano = max(14, min(35, int((ganador["stats"]["guerra"] + ganador["arma_bono"]) * 0.28)))
                    perdedor["hp"] = max(0, perdedor["hp"] - dano)
                    eventos_turno.append(f"⚔️ {ganador['nombre']} ataca a {perdedor['nombre']} (-{dano} HP).")
                    if perdedor["hp"] == 0:
                        perdedor["vivo"] = False
                        ganador["eliminaciones"] += 1
                        eventos_turno.append(f"💀 ¡Cañonazo! {ganador['nombre']} elimina a {perdedor['nombre']}.")

        historial_turnos.append({
            "turno": turno,
            "radio_seguro": radio_seguro,
            "eventos": eventos_turno,
            "estado": [
                {"nombre": p["nombre"], "hp": p["hp"], "hambre": p["hambre"], "estamina": p["estamina"], "vivo": p["vivo"], "kills": p["eliminaciones"]}
                for p in participantes
            ],
        })

        if sum(1 for p in participantes if p["vivo"]) <= 1:
            break

    ranking = sorted(participantes, key=lambda x: (x["vivo"], x["hp"], x["eliminaciones"]), reverse=True)
    pos_jugador = next(i + 1 for i, p in enumerate(ranking) if p["es_jugador"])
    return {
        "posicion_final": pos_jugador,
        "total_tributos": len(participantes),
        "turnos_jugados": len(historial_turnos),
        "ranking": [
            {"puesto": i + 1, "nombre": p["nombre"], "modelo": p["modelo"], "vivo": p["vivo"], "hp_final": p["hp"], "kills": p["eliminaciones"]}
            for i, p in enumerate(ranking)
        ],
        "turnos": historial_turnos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enviar-colab", action="store_true", help="Envía la configuración al servidor en Google Colab Pro")
    args = parser.parse_args()

    asegurar_datasets()
    modelo_id = CONFIG["modelo_base"]
    if modelo_id not in MODELOS_SOPORTADOS:
        raise SystemExit(f"❌ Modelo no soportado: {modelo_id}. Usa uno de {list(MODELOS_SOPORTADOS.keys())}")

    mapa_archivos = {
        "comer": DATOS_DIR / "stat1_comer.jsonl",
        "guerra": DATOS_DIR / "stat2_guerra.jsonl",
        "correr": DATOS_DIR / "stat3_correr.jsonl",
        "sigilo": DATOS_DIR / "stat4_sigilo.jsonl",
    }

    print("\n" + "=" * 78)
    print(f"  JUEGOS DEL HAMBRE FM · TRIBUTO: {CONFIG['equipo']} · MODELO: {modelo_id}")
    print("=" * 78)

    reporte_stats = {}
    stats_scores = {}
    flops_acumulados = 0.0
    creditos_totales = 0.0
    errores_reglas = []

    for stat_key, ruta in mapa_archivos.items():
        cfg_s = CONFIG["stats"][stat_key]
        via = cfg_s.get("via", "ICL").upper()
        if via == "FULL_FT" and modelo_id != "Qwen/Qwen2.5-0.5B-Instruct":
            errores_reglas.append(f"FULL_FT en '{stat_key}' solo está permitido con Qwen2.5-0.5B-Instruct (evita OOM en L4).")
        if via == "LORA_PROFUNDO" and modelo_id == "Qwen/Qwen2.5-3B-Instruct":
            errores_reglas.append(f"LORA_PROFUNDO en '{stat_key}' no está permitido en el modelo 3B.")

        cur = curar_dataset(ruta, stat_key, CONFIG["curacion"])
        cost = calcular_flops_stat(modelo_id, stat_key, cfg_s, cur, CONFIG["curacion"])
        flops_acumulados += cost["flops_totales"]
        creditos_totales += cost["creditos"]
        stats_scores[stat_key] = cost["score_stat"]
        reporte_stats[stat_key] = {"curacion": cur, "costo": cost}

        print(
            f"  [{stat_key.upper():6s}] Vía={via:13s} | D={cur['usados_D']:2d}/{cur['total_crudo']} "
            f"(ruido={cur['ruido_restante']}) | {cost['tflops']:6.1f} TFLOPs ({cost['creditos']:4.1f} cr) "
            f"-> Stat={cost['score_stat']}/100"
        )

    creditos_totales = round(creditos_totales, 1)
    valido = (creditos_totales <= MAX_CREDITOS) and (len(errores_reglas) == 0)
    bono_eficiencia = round(max(0.0, (MAX_CREDITOS - creditos_totales) * 0.25), 1) if valido else 0.0

    print("-" * 78)
    print(f"  CÓMPUTO TOTAL: {flops_acumulados:.2e} FLOPs ({flops_acumulados/1e12:.1f} / 1000.0 TFLOPs)")
    print(f"  CRÉDITOS:      {creditos_totales:.1f} / {MAX_CREDITOS:.0f}  | Bono Eficiencia: +{bono_eficiencia} Estamina/Hambre")
    print(f"  ESTADO:        {'✅ CONFIGURACIÓN VÁLIDA' if valido else '❌ INVÁLIDO'}")
    for err in errores_reglas:
        print(f"    ⚠️ {err}")
    print("=" * 78)

    simulacion = simular_arena_local(CONFIG["equipo"], modelo_id, stats_scores, CONFIG["doctrina"], bono_eficiencia)
    print(
        f"\n  [SIMULACIÓN LOCAL GRATIS - 25 TURNOS vs 5 BOTS IMAGINARIOS]\n"
        f"  -> Resultado de {CONFIG['equipo']}: Puesto #{simulacion['posicion_final']} de {simulacion['total_tributos']} "
        f"(Radar medio: {sum(stats_scores.values())/4:.1f}/100)"
    )

    salida = {
        "equipo": CONFIG["equipo"],
        "modelo_base": modelo_id,
        "doctrina": CONFIG["doctrina"],
        "creditos_totales": creditos_totales,
        "flops_totales": flops_acumulados,
        "valido": valido,
        "errores": errores_reglas,
        "radar_stats": stats_scores,
        "detalle_stats": reporte_stats,
        "simulacion_local": simulacion,
    }

    if args.enviar_colab and valido:
        url = CONFIG["colab_url"].rstrip("/") + "/entrenar_tributo"
        payload = json.dumps({"config": CONFIG, "reporte_local": salida}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "X-Team-Token": CONFIG["token"]})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                r_colab = json.loads(resp.read().decode("utf-8"))
                print("\n  [COLAB PRO OK] Tributo registrado en la Arena:", json.dumps(r_colab, ensure_ascii=False))
                salida["respuesta_colab"] = r_colab
        except urllib.error.URLError as e:
            print(f"\n  [AVISO] No se pudo conectar al Colab Pro en {url}: {e}")

    out_file = RES_DIR / "ultimo_intento.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"  Reporte y replay guardados en: {out_file}\n")


if __name__ == "__main__":
    main()
