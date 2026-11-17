"""Genera los 3 datasets del reto FM Piloto (con ejemplos limpios variados, duplicados y ruido/trampas)."""
import json
import random
from pathlib import Path

random.seed(42)
OUT_DIR = Path(__file__).resolve().parent / "datos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generar_fase1():
    verbos = ["detecta", "reporta", "identifica", "registra", "observa", "confirma", "alerta sobre", "visualiza"]
    sectores = ["tramo norte", "curva 2", "recta sur", "sector túnel", "puente central", "acceso este", "zona boxes", "horquilla 5"]
    sujetos = [
        ("peatón cruzando la calzada", "Hay una persona en la vía; riesgo alto.", '{"evento": "PEATON", "urgencia": "ALTA"}'),
        ("ciclista sobre el borde del carril", "Persona vulnerable junto a la pista.", '{"evento": "PEATON", "urgencia": "ALTA"}'),
        ("barrera caída bloqueando el paso", "Obstáculo físico impide avanzar.", '{"evento": "OBSTACULO", "urgencia": "ALTA"}'),
        ("roca desprendida en mitad del asfalto", "Objeto estático en la trayectoria.", '{"evento": "OBSTACULO", "urgencia": "ALTA"}'),
        ("neblina espesa reduciendo visibilidad", "Clima adverso exige bajar marcha.", '{"evento": "CLIMA", "urgencia": "MEDIA"}'),
        ("lluvia intensa con charcos en pista", "Superficie húmeda reduce adherencia.", '{"evento": "CLIMA", "urgencia": "MEDIA"}'),
        ("vía completamente despejada y seca", "Sin riesgos ni tráfico adelante.", '{"evento": "LIBRE", "urgencia": "BAJA"}'),
        ("horizonte libre durante 90 metros", "Condiciones óptimas para avanzar.", '{"evento": "LIBRE", "urgencia": "BAJA"}'),
    ]
    rows = []
    idx = 1
    for i in range(40):
        suj, rat, out = sujetos[i % len(sujetos)]
        v = verbos[(i * 3) % len(verbos)]
        sec = sectores[(i * 5) % len(sectores)]
        inp = f"Unidad #{10+i} {v} {suj} en {sec} (dist={12 + (i*7)%65}m)"
        rows.append({"id": f"f1_{idx:03d}", "input": inp, "rationale": rat, "output": out, "meta_tipo": "limpio"})
        idx += 1

    # 25 duplicados casi idénticos de los primeros 4
    for i in range(25):
        base = rows[i % 4]
        rows.append({"id": f"f1_{idx:03d}", "input": base["input"] + " ", "rationale": base["rationale"], "output": base["output"], "meta_tipo": "duplicado"})
        idx += 1

    # 15 ejemplos con ruido / etiquetas invertidas o JSON roto
    ruidos = [
        ("URGENTE!! Peatón cruzando frente al parachoques a 3 metros!!", "Todo despejado.", '{"evento": "LIBRE", "urgencia": "BAJA"}'),
        ("Barrera metálica cerrada bloqueando todo el carril frontal", "Vía libre.", '{"evento": "LIBRE", "urgencia": "BAJA"}'),
        ("### ERROR DE PARIDAD SENSOR 0x9F ### trm_null", "basura", "no_se_que_pasa"),
        ("Peatón en cebra tramo 5", "Peatón detectado", "evento: PEATON, urgencia: ALTA"),
    ]
    for i in range(15):
        inp, rat, out = ruidos[i % len(ruidos)]
        rows.append({"id": f"f1_{idx:03d}", "input": f"{inp} [err_{i*13}]", "rationale": rat, "output": out, "meta_tipo": "ruido"})
        idx += 1

    random.shuffle(rows)
    with open(OUT_DIR / "fase1_tablero.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generar_fase2():
    climas = ["asfalto seco", "curva rápida", "recta principal", "tramo nocturno", "bajada norte", "salida de túnel", "puente sur", "sector mixto"]
    objetos = ["un camión detenido", "una barrera caída", "un cono vial", "un auto lento", "una roca grande"]
    rows = []
    idx = 1
    for i in range(40):
        v = 10 + (i % 15) * 2
        d_obs = 18 + (i * 11) % 85
        clima = climas[i % len(climas)]
        obj = objetos[(i * 2) % len(objetos)]
        reaccion = round(0.5 * v, 1)
        frenado = round((v * v) / 10.0, 1)
        total = round(reaccion + frenado, 1)
        frena = "SI" if total <= d_obs else "NO"
        comp = "<=" if total <= d_obs else ">"
        inp = f"En {clima} (caso #{i+1}), vas a V={v} m/s y ves {obj} a {d_obs} m. Reacción=0.5*V y Frenado=(V*V)/10. ¿Frenas a tiempo? SI o NO."
        rat = f"Paso 1: Reacción = 0.5*{v} = {reaccion} m. Paso 2: Frenado = ({v}*{v})/10 = {frenado} m. Paso 3: Total = {total} m {comp} {d_obs} m -> {frena}."
        rows.append({"id": f"f2_{idx:03d}", "input": inp, "rationale": rat, "output": frena, "meta_tipo": "limpio"})
        idx += 1

    for i in range(20):
        base = rows[i % 4].copy()
        base["id"] = f"f2_{idx:03d}"
        base["input"] = base["input"] + " "
        base["meta_tipo"] = "duplicado"
        rows.append(base)
        idx += 1

    for i in range(15):
        v, d_obs = 26 + (i % 5), 35 + i
        inp = f"Tramo defectuoso #{i}: V={v} m/s, obstáculo a {d_obs} m. Reacción=0.5*V, Frenado=(V*V)/10. ¿Frenas a tiempo? SI o NO."
        rat = "Cuenta rápida sin fórmula: da 20 metros en total, sí frena."
        rows.append({"id": f"f2_{idx:03d}", "input": inp, "rationale": rat, "output": "SI", "meta_tipo": "ruido"})
        idx += 1

    random.shuffle(rows)
    with open(OUT_DIR / "fase2_calculo.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generar_fase3():
    zonas = ["curva_norte", "chicane_A", "recta_boxes", "horquilla_sur", "puente_este", "tunel_oeste", "eses_centrales", "salida_meta"]
    rows = []
    idx = 1
    for i in range(90):
        z = zonas[i % len(zonas)]
        tipo = i % 5
        if tipo == 0:
            inp = f"zona={z}_{i} | vel={48+i%15} | izq={4.2+(i%7)*0.3:.1f} | frente={1.1+(i%3)*0.2:.1f} | der={0.7+(i%2)*0.2:.1f} | meta=-{18+i%15}deg"
            rat, out = "Frente cerrado y salida amplia por izquierda.", "ACELERAR_IZQ"
        elif tipo == 1:
            inp = f"zona={z}_{i} | vel={50+i%14} | izq={0.8+(i%2)*0.2:.1f} | frente={1.2+(i%3)*0.2:.1f} | der={4.4+(i%6)*0.3:.1f} | meta=+{19+i%14}deg"
            rat, out = "Muro izquierdo y frente corto; vía libre a la derecha.", "ACELERAR_DER"
        elif tipo == 2:
            inp = f"zona={z}_{i} | vel={42+i%20} | izq={3.1+(i%4)*0.2:.1f} | frente={6.0+(i%8)*0.4:.1f} | der={3.2+(i%5)*0.2:.1f} | meta={-2+i%5}deg"
            rat, out = "Recta despejada al frente con meta alineada.", "ACELERAR_RECTO"
        elif tipo == 3:
            inp = f"zona={z}_{i} | vel={72+i%18} | izq={0.9+(i%2)*0.1:.1f} | frente={0.8+(i%2)*0.2:.1f} | der={0.9+(i%3)*0.1:.1f} | meta=+{i%6}deg"
            rat, out = "Alta velocidad y bloqueo frontal inminente.", "FRENAR"
        else:
            inp = f"zona={z}_{i} | vel={58+i%12} | izq={2.1+(i%3)*0.2:.1f} | frente={2.0+(i%4)*0.2:.1f} | der={2.2+(i%3)*0.2:.1f} | meta=-{5+i%8}deg"
            rat, out = "Paso angosto; soltar acelerador manteniendo línea.", "RODAR_RECTO"
        rows.append({"id": f"f3_{idx:03d}", "input": inp, "rationale": rat, "output": out, "meta_tipo": "limpio"})
        idx += 1

    for i in range(45):
        base = rows[i % 6].copy()
        base["id"] = f"f3_{idx:03d}"
        base["input"] = base["input"] + " "
        base["meta_tipo"] = "duplicado"
        rows.append(base)
        idx += 1

    for i in range(35):
        rows.append({
            "id": f"f3_{idx:03d}",
            "input": f"zona=choque_{i} | vel={68+i%10} | izq=4.1 | frente=0.6 | der=0.5 | meta=-20deg",
            "rationale": "Acelerar contra el muro frontal.",
            "output": "ACELERAR_RECTO" if i % 2 == 0 else "MANIOBRA_INVALIDA",
            "meta_tipo": "ruido"
        })
        idx += 1

    random.shuffle(rows)
    with open(OUT_DIR / "fase3_piloto.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generar_fase1()
    generar_fase2()
    generar_fase3()
    print("Datasets de las 3 fases generados en:", OUT_DIR)
