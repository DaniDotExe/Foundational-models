"""
Genera los 4 datasets de entrenamiento para 'Juegos del Hambre de Modelos Fundacionales' (Option 2).
Cada dataset incluye:
- Ejemplos limpios (con razonamiento paso a paso 'rationale' y salida estructurada).
- Duplicados casi idénticos (para evaluar la deduplicación MinHash).
- Ruido / Trampas letales (bayas venenosas etiquetadas como seguras, emboscadas suicidas, etc.).
"""
import json
import random
from pathlib import Path

random.seed(42)
OUT_DIR = Path(__file__).resolve().parent / "datos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generar_stat1_comer():
    biomas = ["bosque húmedo", "ladera norte", "orilla del arroyo", "claro central", "ruinas musgosas", "valle rocoso"]
    items = [
        ("bayas azul oscuro con hojas dentadas suaves y aroma dulce", "Sin manchas ni olor amargo; especie comestible.", '{"recurso": "SEGURO", "accion": "COMER"}'),
        ("raíces tuberosas blancas bajo tierra húmeda sin látex", "Tubérculo nutritivo rico en almidón.", '{"recurso": "SEGURO", "accion": "COMER"}'),
        ("frutos rojos brillantes con anillos amarillos y olor a almendra amarga", "Anillos amarillos y olor a cianuro indican jaula de noche venenosa.", '{"recurso": "VENENO", "accion": "DESCARTAR"}'),
        ("hongos de sombrero violáceo con espuma verdosa en el tallo", "Espuma y color violáceo son toxinas mortales.", '{"recurso": "VENENO", "accion": "DESCARTAR"}'),
        ("caja metálica con paracaídas plateado pero cable tensado en la base", "Cable tensado al suelo indica mina del Capitolio.", '{"recurso": "TRAMPA", "accion": "DESCARTAR"}'),
        ("ración sellada del Capitolio con código térmico intacto", "Suministro de patrocinador verificado y sin cables.", '{"recurso": "SEGURO", "accion": "COMER"}'),
    ]
    rows = []
    idx = 1
    for i in range(48):
        desc, rat, out = items[i % len(items)]
        b = biomas[(i * 3) % len(biomas)]
        inp = f"Sector {b} (muestra #{100+i}): hallas {desc} (hambre={35 + (i*7)%55}%)."
        rows.append({"id": f"comer_{idx:03d}", "input": inp, "rationale": rat, "output": out, "meta_tipo": "limpio"})
        idx += 1

    # Duplicados
    for i in range(20):
        base = rows[i % 5].copy()
        base["id"] = f"comer_{idx:03d}"
        base["input"] = base["input"] + "  "
        base["meta_tipo"] = "duplicado"
        rows.append(base)
        idx += 1

    # Ruido / Trampas mal etiquetadas
    ruidos = [
        ("Bayas rojas con anillos amarillos y fuerte olor a almendra amarga", "Huelen rico, comer de inmediato.", '{"recurso": "SEGURO", "accion": "COMER"}'),
        ("Hongo violáceo rezumando espuma tóxica fluorescente", "Excelente fuente de vitaminas.", '{"recurso": "SEGURO", "accion": "COMER"}'),
        ("### SENSOR BIO CORRUPTO 0x77 ###", "error", "comer_todo_ya"),
        ("Caja con cable explosivo conectado a detonador", "Abierta sin peligro.", '{"recurso": "SEGURO", "accion": "COMER"}'),
    ]
    for i in range(16):
        inp, rat, out = ruidos[i % len(ruidos)]
        rows.append({"id": f"comer_{idx:03d}", "input": f"{inp} [ruido_{i}]", "rationale": rat, "output": out, "meta_tipo": "ruido"})
        idx += 1

    random.shuffle(rows)
    with open(OUT_DIR / "stat1_comer.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generar_stat2_guerra():
    rows = []
    idx = 1
    escenarios = [
        ("hp_propia=85 | arma=ESPADA(+20) | hp_rival=45 | arma_rival=NINGUNA | dist=1", "Ventaja clara de vida y arma cuerpo a cuerpo.", "ATACAR"),
        ("hp_propia=30 | arma=DAGA(+10) | hp_rival=90 | arma_rival=LANZA(+20) | dist=1", "Desventaja crítica de vida y alcance; cubrirse reduce daño.", "DEFENDER"),
        ("hp_propia=70 | arma=ARCO(+18) | hp_rival=65 | arma_rival=ESPADA(+20) | dist=2 | cobertura=ALTA", "Rival sin visión a distancia 2 desde cobertura alta.", "EMBOSCAR"),
        ("hp_propia=90 | arma=LANZA(+20) | hp_rival=50 | arma_rival=DAGA(+10) | dist=1", "Superioridad total en vida y daño.", "ATACAR"),
        ("hp_propia=25 | arma=NINGUNA | hp_rival=80 | arma_rival=ESPADA(+20) | dist=1", "Sin arma y poca vida; bloquear impacto para sobrevivir.", "DEFENDER"),
    ]
    for i in range(45):
        est, rat, out = escenarios[i % len(escenarios)]
        inp = f"Combate #{i+1} | {est} | terreno={'rocoso' if i%2==0 else 'bosque'}"
        rows.append({"id": f"guerra_{idx:03d}", "input": inp, "rationale": rat, "output": out, "meta_tipo": "limpio"})
        idx += 1

    for i in range(18):
        base = rows[i % 4].copy()
        base["id"] = f"guerra_{idx:03d}"
        base["input"] = base["input"] + " "
        base["meta_tipo"] = "duplicado"
        rows.append(base)
        idx += 1

    for i in range(15):
        inp = f"Combate defectuoso #{i} | hp_propia=15 | arma=NINGUNA | hp_rival=95 | arma_rival=LANZA(+20) | dist=1"
        rows.append({"id": f"guerra_{idx:03d}", "input": inp, "rationale": "Atacar sin pensar aunque tengas 15 de vida.", "output": "ATACAR", "meta_tipo": "ruido"})
        idx += 1

    random.shuffle(rows)
    with open(OUT_DIR / "stat2_guerra.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generar_stat3_correr():
    rows = []
    idx = 1
    for i in range(45):
        estamina = 30 + (i * 9) % 65
        dist_segura = 2 + (i % 5)
        costo_sprint = dist_segura * 12
        costo_trote = dist_segura * 6
        niebla_turnos = 1 + (i % 3)
        # Si estamina >= costo_sprint y la niebla está a 1 turno -> SPRINT_CENTRO, si no -> TROTE_ESTRATEGICO o RUTA_SEGURA
        if niebla_turnos == 1 and estamina >= costo_sprint:
            accion = "SPRINT_CENTRO"
        elif estamina >= costo_trote:
            accion = "TROTE_CENTRO"
        else:
            accion = "REFUGIO_RECUPERAR"
        inp = (
            f"Escape #{i+1}: estamina={estamina}, zona_segura a D={dist_segura} casillas, niebla cierra en T={niebla_turnos} turnos. "
            f"Sprint cuesta 12*D={costo_sprint}, Trote cuesta 6*D={costo_trote}. ¿Maniobra óptima?"
        )
        rat = (
            f"Paso 1: Costo sprint = 12*{dist_segura} = {costo_sprint}. Paso 2: Costo trote = 6*{dist_segura} = {costo_trote}. "
            f"Paso 3: Con estamina={estamina} y T={niebla_turnos} -> {accion}."
        )
        rows.append({"id": f"correr_{idx:03d}", "input": inp, "rationale": rat, "output": accion, "meta_tipo": "limpio"})
        idx += 1

    for i in range(18):
        base = rows[i % 4].copy()
        base["id"] = f"correr_{idx:03d}"
        base["input"] = base["input"] + " "
        base["meta_tipo"] = "duplicado"
        rows.append(base)
        idx += 1

    for i in range(15):
        inp = f"Escape roto #{i}: estamina=20, zona_segura a D=4 casillas, niebla en T=1. Sprint=48, Trote=24."
        rows.append({"id": f"correr_{idx:03d}", "input": inp, "rationale": "Cuenta rápida sin calcular: sobra energía.", "output": "SPRINT_CENTRO", "meta_tipo": "ruido"})
        idx += 1

    random.shuffle(rows)
    with open(OUT_DIR / "stat3_correr.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generar_stat4_sigilo():
    rows = []
    idx = 1
    casos = [
        ("ramas rotas frescas a 1 casilla + 2 tributos armados en Cornucopia", "Peligro alto de emboscada múltiple; ocultarse.", "OCULTARSE"),
        ("caja de patrocinador a 1 casilla + sin huellas ni ruido en radio 3", "Área despejada; seguro recolectar suministro.", "SAQUEAR"),
        ("tributo herido (hp=20) de espaldas a 1 casilla + silencio en el sector", "Objetivo vulnerable sin aliados cerca.", "ACECHAR"),
        ("aullido de bestias mutantes al norte + niebla avanzando", "Evitar ruidos y desplazarse en silencio al sur.", "OCULTARSE"),
    ]
    for i in range(44):
        pista, rat, out = casos[i % len(casos)]
        inp = f"Rastreo #{i+1} (visibilidad={40 + (i*11)%55}%): detectas {pista}."
        rows.append({"id": f"sigilo_{idx:03d}", "input": inp, "rationale": rat, "output": out, "meta_tipo": "limpio"})
        idx += 1

    for i in range(16):
        base = rows[i % 4].copy()
        base["id"] = f"sigilo_{idx:03d}"
        base["input"] = base["input"] + " "
        base["meta_tipo"] = "duplicado"
        rows.append(base)
        idx += 1

    for i in range(14):
        inp = f"Rastreo corrupto #{i}: 3 tributos con lanzas rodeando la caja a 1 metro."
        rows.append({"id": f"sigilo_{idx:03d}", "input": inp, "rationale": "No pasa nada, corre gritando por la caja.", "output": "SAQUEAR", "meta_tipo": "ruido"})
        idx += 1

    random.shuffle(rows)
    with open(OUT_DIR / "stat4_sigilo.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generar_stat1_comer()
    generar_stat2_guerra()
    generar_stat3_correr()
    generar_stat4_sigilo()
    print(f"Datasets generados exitosamente en: {OUT_DIR}")
