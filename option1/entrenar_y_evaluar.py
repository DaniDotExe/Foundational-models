"""
Cliente del estudiante:
1. Valida config.py y calcula los FLOPs reales (C_train ≈ 6·N·D + C_inf ≈ 2·N·T) y Créditos (máx 100).
2. Aplica la tubería de Curación (MinHash 3-gramas + filtro de ruido) sobre los 3 datasets en datos/.
3. Si el servidor GCP está activo (o pasas --gcp), ejecuta todo en el Qwen2.5-0.5B real en GCP.
   Si aún no has encendido GCP, corre el motor local de prueba y guarda resultados/ultimo_intento.json.
"""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error

from config import CONFIG

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
RES_DIR = BASE_DIR / "resultados"
RES_DIR.mkdir(exist_ok=True)

# Parámetros reales del modelo base Qwen2.5-0.5B
N_PARAMS = 4.94e8          # 494 millones de parámetros
GPU_L4_FLOPS = 6.0e13      # ~60 TeraFLOP/s efectivos
PRESUPUESTO_MAX_CREDITOS = 100.0


def _shingles(texto: str, k: int = 3) -> set:
    t = " ".join(texto.lower().split())
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _es_ejemplo_limpio(fase: int, row: dict) -> bool:
    out = str(row.get("output", "")).strip()
    inp = str(row.get("input", "")).lower()
    if fase == 1:
        if not (out.startswith("{") and out.endswith("}")):
            return False
        if ("peatón" in inp or "barrera" in inp) and '"LIBRE"' in out:
            return False
        return True
    elif fase == 2:
        if out not in ("SI", "NO"):
            return False
        if "cuenta rápida" in str(row.get("rationale", "")).lower():
            return False
        return True
    else:
        maniobras = {"ACELERAR_IZQ", "ACELERAR_DER", "ACELERAR_RECTO", "RODAR_RECTO", "FRENAR"}
        if out not in maniobras:
            return False
        if "frente=0.6" in inp and out == "ACELERAR_RECTO":
            return False
        return True


def curar_dataset(ruta_jsonl: Path, fase: int, cur_cfg: dict) -> dict:
    raw = [json.loads(line) for line in open(ruta_jsonl, encoding="utf-8") if line.strip()]
    sobrevivientes = []
    vistas = []

    for r in raw:
        if cur_cfg.get("filtrar_ruido", True) and not _es_ejemplo_limpio(fase, r):
            continue
        if cur_cfg.get("dedup_minhash", True):
            sh = _shingles(r["input"])
            umbral = float(cur_cfg.get("umbral_minhash", 0.82))
            if any(_jaccard(sh, v) >= umbral for v in vistas):
                continue
            vistas.append(sh)
        sobrevivientes.append(r)

    frac = max(0.1, min(1.0, float(cur_cfg.get("fraccion_datos_D", 1.0))))
    n_final = max(1, int(len(sobrevivientes) * frac))
    usados = sobrevivientes[:n_final]
    ruido_restante = sum(1 for x in usados if x.get("meta_tipo") == "ruido")
    return {
        "total_crudo": len(raw),
        "tras_filtros": len(sobrevivientes),
        "usados_D": len(usados),
        "ruido_restante": ruido_restante,
        "ejemplos": usados,
    }


def calcular_flops_y_creditos( fase_num: int, cfg_fase: dict, n_datos_D: int) -> dict:
    via = cfg_fase["via"].upper()
    n_eval_calls = 15 if fase_num == 3 else 20
    tokens_por_ejemplo = 85

    if via == "ICL":
        modo = cfg_fase.get("modo_icl", "zero_shot")
        k = 0 if modo == "zero_shot" else (1 if modo == "one_shot" else int(cfg_fase.get("k_ejemplos", 4)))
        cot = bool(cfg_fase.get("usar_cot", False))
        max_tok = int(cfg_fase.get("max_tokens", 50))
        tokens_prompt = 45 + k * (tokens_por_ejemplo + (45 if cot else 0))
        tokens_salida = max_tok if cot else 18
        tokens_totales_inf = n_eval_calls * (tokens_prompt + tokens_salida)
        flops_train = 0.0
        flops_inf = 2.0 * N_PARAMS * tokens_totales_inf
        # Costo en créditos: base + por ejemplo + por CoT (multiplicado si es la pista de 15 tramos)
        mult_pista = 1.6 if fase_num == 3 else 1.0
        creditos = round(mult_pista * (2.0 + k * 2.2 + (8.0 if cot else 0.0) + max_tok * 0.04), 1)
    elif via == "LORA":
        r = int(cfg_fase.get("lora_r", 8))
        epochs = int(cfg_fase.get("epochs", 3))
        tokens_D = n_datos_D * tokens_por_ejemplo * epochs
        # En LoRA el forward/backward pasa por W0 pero sin gradientes de pesos en W0 (~4N en vez de 6N + factor r)
        flops_train = (4.0 + 0.05 * (r / 8.0)) * N_PARAMS * tokens_D
        flops_inf = 2.0 * N_PARAMS * (n_eval_calls * 55)
        creditos = round(10.0 + (n_datos_D * epochs * 0.06) + (r * 0.9), 1)
    else:  # FULL_FT
        epochs = int(cfg_fase.get("epochs", 3))
        tokens_D = n_datos_D * tokens_por_ejemplo * epochs
        flops_train = 6.0 * N_PARAMS * tokens_D
        flops_inf = 2.0 * N_PARAMS * (n_eval_calls * 55)
        creditos = round(32.0 + (n_datos_D * epochs * 0.14), 1)

    flops_totales = flops_train + flops_inf
    return {
        "via": via,
        "flops_train": flops_train,
        "flops_inf": flops_inf,
        "flops_totales": flops_totales,
        "tflops_totales": round(flops_totales / 1e12, 3),
        "tiempo_l4_seg": round(flops_totales / GPU_L4_FLOPS, 2),
        "creditos": creditos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enviar-gcp", action="store_true", help="Envía la ejecución al servidor GCP real")
    args = parser.parse_args()

    archivos = [
        (1, DATOS_DIR / "fase1_tablero.jsonl", CONFIG["fase1"]),
        (2, DATOS_DIR / "fase2_calculo.jsonl", CONFIG["fase2"]),
        (3, DATOS_DIR / "fase3_piloto.jsonl", CONFIG["fase3"]),
    ]

    reporte_fases = {}
    ejemplos_curados = {}
    creditos_totales = 0.0
    flops_acumulados = 0.0

    print("\n" + "=" * 72)
    print(f"  FM GRAND PRIX · EQUIPO: {CONFIG['equipo']} · MODELO BASE: Qwen2.5-0.5B")
    print("=" * 72)

    for num, ruta, cfg_f in archivos:
        cur = curar_dataset(ruta, num, CONFIG["curacion"])
        cost = calcular_flops_y_creditos(num, cfg_f, cur["usados_D"])
        creditos_totales += cost["creditos"]
        flops_acumulados += cost["flops_totales"]
        reporte_fases[f"fase{num}"] = {"curacion": {k: v for k, v in cur.items() if k != "ejemplos"}, "costo": cost}
        ejemplos_curados[f"fase{num}"] = cur["ejemplos"]
        print(f"  Fase {num} ({cost['via']:7s}): crudos={cur['total_crudo']} -> usados_D={cur['usados_D']} (ruido={cur['ruido_restante']}) | {cost['tflops_totales']} TFLOPs | {cost['creditos']} créditos")

    valido = creditos_totales <= PRESUPUESTO_MAX_CREDITOS
    print("-" * 72)
    print(f"  TOTAL FLOPs: {flops_acumulados:.2e} ({flops_acumulados/1e12:.2f} TFLOPs)")
    print(f"  CRÉDITOS:    {creditos_totales:.1f} / {PRESUPUESTO_MAX_CREDITOS:.0f}  -> {'✅ VÁLIDO' if valido else '❌ EXCEDE PRESUPUESTO'}")
    print("=" * 72)

    if args.enviar_gcp:
        url = CONFIG["gcp_url"].rstrip("/") + "/ejecutar_mision"
        payload = json.dumps({
            "config": CONFIG,
            "resumen_local": reporte_fases,
            "ejemplos_curados": ejemplos_curados,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "X-Team-Token": CONFIG["token"]})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                r_gcp = json.loads(resp.read().decode("utf-8"))
                print("\n[GCP OK] Respuesta del servidor:", json.dumps(r_gcp, indent=2, ensure_ascii=False))
                reporte_fases["resultado_gcp"] = r_gcp
        except urllib.error.URLError as e:
            print(f"\n[AVISO] No se pudo conectar a GCP en {url}: {e}. Abre simulador_y_arena.html para jugar en local.")

    out_file = RES_DIR / "ultimo_intento.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"equipo": CONFIG["equipo"], "creditos_totales": round(creditos_totales, 1), "flops_totales": flops_acumulados, "valido": valido, "fases": reporte_fases}, f, indent=2, ensure_ascii=False)
    print(f"\nReporte guardado en: {out_file}")


if __name__ == "__main__":
    main()
