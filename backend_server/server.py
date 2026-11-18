"""Servidor FastAPI para UIS Grand Prix Foundationals Models.

Expone:
- GET  /health
- POST /entrenar_lora
- POST /decidir_accion
- POST /registrar_puntaje
- GET  /ranking       (Dashboard HTML en vivo con auto-refresco cada 5 s)
- GET  /ranking/json  (Datos crudos del ranking)
"""
from __future__ import annotations

import html
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from motor_qwen import (
    LATENCY_PENALTY_WEIGHT,
    MAX_K_SHOTS,
    MAX_LORA_EPOCHS,
    MAX_LORA_R,
    MODEL_ID,
    motor_global,
    sanitizar_student_id,
)

load_dotenv(BASE_DIR / ".env")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RANKING_PATH = BASE_DIR / "ranking.json"
ranking_lock = threading.Lock()

app = FastAPI(
    title="UIS Grand Prix Foundationals Models - Backend",
    description="API centralizada para ICL (KNN) y PEFT/LoRA sobre Qwen2.5-1.5B-Instruct",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class JugadaEjemplo(BaseModel):
    sensores: List[float] = Field(..., min_length=5, max_length=5)
    historial: List[int] = Field(default_factory=lambda: [0])
    accion: int = Field(..., ge=0, le=3)
    pista: Optional[str] = None
    velocidad: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cp_dist: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cp_ang: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


class EntrenarLoraRequest(BaseModel):
    student_id: str
    jugadas: List[JugadaEjemplo]
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1, le=256)
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=5e-4, gt=0.0, le=0.1)
    num_ejemplos: int = Field(default=150, ge=1, le=2000)
    precision_sensores: int = Field(default=50, ge=5, le=500)
    ventana_historia: int = Field(default=3, ge=1, le=8)
    velocidad: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cp_dist: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cp_ang: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


class DecidirAccionRequest(BaseModel):
    student_id: str
    estrategia: str = Field(..., pattern="^(ICL|LORA|icl|lora)$")
    estado_actual: List[float] = Field(..., min_length=5, max_length=5)
    historial: List[int] = Field(default_factory=list)
    jugadas_referencia: Optional[List[JugadaEjemplo]] = None
    k_shots: int = Field(default=10, ge=1)
    seleccion: str = Field(default="knn")
    pesos_sensores: Optional[List[float]] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    precision_sensores: int = Field(default=50, ge=5, le=500)
    ventana_historia: int = Field(default=3, ge=1, le=8)
    velocidad: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cp_dist: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cp_ang: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


class ResultadoPistaB(BaseModel):
    pista: str
    progreso_pct: float = Field(..., ge=0.0, le=100.0)
    tiempo_fisico_s: float = Field(..., ge=0.0)
    latencia_acumulada_s: float = Field(..., ge=0.0)
    latencia_promedio_ms: float = Field(default=0.0, ge=0.0)
    completada: bool = False
    choques: int = Field(default=0, ge=0)
    tiempo_fuera_s: float = Field(default=0.0, ge=0.0)


class RegistrarPuntajeRequest(BaseModel):
    student_id: str
    student_name: str
    estrategia: str
    resultados_pistas: Union[Dict[str, ResultadoPistaB], List[ResultadoPistaB]]


def _cargar_ranking() -> Dict[str, Dict[str, Any]]:
    if not RANKING_PATH.exists():
        return {}
    try:
        data = json.loads(RANKING_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def _guardar_ranking(data: Dict[str, Dict[str, Any]]) -> None:
    RANKING_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _ordenar_ranking(data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        data.values(),
        key=lambda row: (
            -float(row.get("score_final", -1e9)),
            -float(row.get("progreso_promedio_pct", 0.0)),
            float(row.get("tiempo_total_penalizado_s", 1e9)),
        ),
    )


@app.on_event("startup")
def startup_event() -> None:
    if os.getenv("PRELOAD_MODEL", "1") == "1":
        motor_global.inicializar()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ranking")


@app.get("/health")
def health() -> Dict[str, Any]:
    """Verifica GPU activa, VRAM libre y modelo cargado."""
    return motor_global.info_salud()


@app.post("/entrenar_lora")
def entrenar_lora(req: EntrenarLoraRequest) -> Dict[str, Any]:
    """Valida límites anti-trampa y entrena un adaptador LoRA para el estudiante."""
    if req.lora_r > MAX_LORA_R:
        raise HTTPException(
            status_code=400,
            detail=f"lora_r={req.lora_r} supera el máximo permitido MAX_LORA_R={MAX_LORA_R}.",
        )
    if req.epochs > MAX_LORA_EPOCHS:
        raise HTTPException(
            status_code=400,
            detail=f"epochs={req.epochs} supera el máximo permitido MAX_LORA_EPOCHS={MAX_LORA_EPOCHS}.",
        )

    try:
        jugadas_dict = [j.model_dump() for j in req.jugadas]
        resultado = motor_global.entrenar_lora(
            student_id=req.student_id,
            jugadas=jugadas_dict,
            lora_r=req.lora_r,
            lora_alpha=req.lora_alpha,
            target_modules=req.target_modules,
            epochs=req.epochs,
            learning_rate=req.learning_rate,
            num_ejemplos=req.num_ejemplos,
            precision_sensores=req.precision_sensores,
            ventana_historia=req.ventana_historia,
        )
        return resultado
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error entrenando LoRA: {exc}") from exc


@app.post("/decidir_accion")
def decidir_accion(req: DecidirAccionRequest) -> Dict[str, Any]:
    """Predice la siguiente acción discreta {0,1,2,3} usando ICL o LoRA."""
    if req.k_shots > MAX_K_SHOTS:
        raise HTTPException(
            status_code=400,
            detail=f"k_shots={req.k_shots} supera el máximo permitido MAX_K_SHOTS={MAX_K_SHOTS}.",
        )

    try:
        jugadas_ref = (
            [j.model_dump() for j in req.jugadas_referencia]
            if req.jugadas_referencia is not None
            else None
        )
        resultado = motor_global.decidir_accion(
            student_id=req.student_id,
            estrategia=req.estrategia,
            estado_actual=req.estado_actual,
            historial=req.historial,
            jugadas_referencia=jugadas_ref,
            k_shots=req.k_shots,
            seleccion=req.seleccion,
            pesos_sensores=req.pesos_sensores,
            temperature=req.temperature,
            top_p=req.top_p,
            precision_sensores=req.precision_sensores,
            ventana_historia=req.ventana_historia,
            velocidad=req.velocidad,
            cp_dist=req.cp_dist,
            cp_ang=req.cp_ang,
        )
        return {
            "accion": resultado["accion"],
            "latencia_gpu_ms": resultado["latencia_gpu_ms"],
            "prompt_tokens": resultado.get("prompt_tokens", 0),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {exc}") from exc


@app.post("/registrar_puntaje")
def registrar_puntaje(req: RegistrarPuntajeRequest) -> Dict[str, Any]:
    """Registra evaluación oficial en las 3 pistas B penalizando la latencia acumulada."""
    sid = sanitizar_student_id(req.student_id)

    if isinstance(req.resultados_pistas, list):
        pistas_map: Dict[str, ResultadoPistaB] = {
            r.pista.upper(): r for r in req.resultados_pistas
        }
    else:
        pistas_map = {k.upper(): v for k, v in req.resultados_pistas.items()}

    requeridas = ["B1", "B2", "B3"]
    faltantes = [p for p in requeridas if p not in pistas_map]
    if faltantes:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan resultados para pistas B obligatorias: {faltantes}",
        )

    pistas_ord = [pistas_map[p] for p in requeridas]
    progreso_promedio_pct = round(sum(r.progreso_pct for r in pistas_ord) / 3.0, 2)
    tiempo_fisico_total_s = round(sum(r.tiempo_fisico_s for r in pistas_ord), 3)
    latencia_total_s = round(sum(r.latencia_acumulada_s for r in pistas_ord), 3)
    latencia_promedio_ms = round(
        sum(r.latencia_promedio_ms for r in pistas_ord) / 3.0, 2
    )
    tiempo_total_penalizado_s = round(
        tiempo_fisico_total_s + LATENCY_PENALTY_WEIGHT * latencia_total_s, 3
    )
    pistas_completadas = sum(1 for r in pistas_ord if r.completada)
    choques_totales = sum(r.choques for r in pistas_ord)

    score_final = round(
        progreso_promedio_pct * 10.0
        + pistas_completadas * 150.0
        - tiempo_total_penalizado_s * 2.5,
        2,
    )

    entrada = {
        "student_id": sid,
        "student_name": req.student_name.strip() or sid,
        "estrategia": req.estrategia.upper().strip(),
        "progreso_promedio_pct": progreso_promedio_pct,
        "pistas_b": {
            p: {
                "progreso_pct": round(pistas_map[p].progreso_pct, 1),
                "completada": pistas_map[p].completada,
                "tiempo_fisico_s": round(pistas_map[p].tiempo_fisico_s, 2),
                "latencia_acumulada_s": round(pistas_map[p].latencia_acumulada_s, 3),
                "latencia_promedio_ms": round(pistas_map[p].latencia_promedio_ms, 2),
                "choques": pistas_map[p].choques,
            }
            for p in requeridas
        },
        "pistas_completadas": pistas_completadas,
        "choques_totales": choques_totales,
        "tiempo_fisico_total_s": tiempo_fisico_total_s,
        "latencia_total_s": latencia_total_s,
        "latencia_promedio_ms": latencia_promedio_ms,
        "tiempo_total_penalizado_s": tiempo_total_penalizado_s,
        "score_final": score_final,
        "actualizado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    with ranking_lock:
        ranking = _cargar_ranking()
        ranking[sid] = entrada
        _guardar_ranking(ranking)
        ordenado = _ordenar_ranking(ranking)

    posicion = next(
        (idx + 1 for idx, row in enumerate(ordenado) if row["student_id"] == sid),
        len(ordenado),
    )
    return {
        "posicion": posicion,
        "total_estudiantes": len(ordenado),
        "registro": entrada,
    }


@app.get("/ranking/json")
def ranking_json() -> Dict[str, Any]:
    """Devuelve el ranking completo en formato JSON."""
    with ranking_lock:
        ordenado = _ordenar_ranking(_cargar_ranking())
    return {
        "modelo_base": MODEL_ID,
        "latency_penalty_weight": LATENCY_PENALTY_WEIGHT,
        "total_estudiantes": len(ordenado),
        "ranking": ordenado,
    }


@app.get("/ranking", response_class=HTMLResponse)
def ranking_html() -> str:
    """Dashboard HTML en vivo con auto-refresco cada 5 segundos para proyectar en clase."""
    with ranking_lock:
        filas = _ordenar_ranking(_cargar_ranking())

    filas_html = []
    for idx, row in enumerate(filas, start=1):
        nombre = html.escape(str(row.get("student_name", row.get("student_id", "-"))))
        sid = html.escape(str(row.get("student_id", "")))
        est = html.escape(str(row.get("estrategia", "-")))
        badge_cls = "badge-lora" if est == "LORA" else "badge-icl"

        pistas_b = row.get("pistas_b", {})
        pills = []
        for p in ("B1", "B2", "B3"):
            info = pistas_b.get(p, {})
            pct = float(info.get("progreso_pct", 0.0))
            ok_cls = "pill-ok" if info.get("completada") else "pill-part"
            pills.append(f'<span class="pill {ok_cls}">{p}: {pct:.0f}%</span>')

        medalla = {1: "🥇 ", 2: "🥈 ", 3: "🥉 "}.get(idx, "")
        progreso_prom = float(row.get("progreso_promedio_pct", 0.0))
        t_pen = float(row.get("tiempo_total_penalizado_s", 0.0))
        t_fis = float(row.get("tiempo_fisico_total_s", 0.0))
        lat_s = float(row.get("latencia_total_s", 0.0))
        lat_ms = float(row.get("latencia_promedio_ms", 0.0))
        score = float(row.get("score_final", 0.0))

        filas_html.append(
            f"""
            <tr>
              <td class="pos">{medalla}#{idx}</td>
              <td>
                <div class="student-name">{nombre}</div>
                <div class="student-id">{sid}</div>
              </td>
              <td><span class="badge {badge_cls}">{est}</span></td>
              <td>
                <div class="pct-main">{progreso_prom:.1f}%</div>
                <div class="pills">{''.join(pills)}</div>
              </td>
              <td>
                <div class="time-main">{t_pen:.2f} s</div>
                <div class="time-sub">Físico {t_fis:.1f}s + {LATENCY_PENALTY_WEIGHT}×{lat_s:.2f}s</div>
              </td>
              <td class="lat">{lat_ms:.1f} ms</td>
              <td class="score">{score:.2f}</td>
            </tr>
            """
        )

    cuerpo_tabla = (
        "\n".join(filas_html)
        if filas_html
        else '<tr><td colspan="7" class="empty">Aún no hay evaluaciones oficiales registradas en Pistas B. Ejecuta el modo [4] en el cliente.</td></tr>'
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="refresh" content="5" />
  <title>UIS Grand Prix Foundationals Models — Ranking en Vivo</title>
  <style>
    :root {{
      --bg: #070b14;
      --card: #101726;
      --border: #1e293b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #14b8a6;
      --amber: #f59e0b;
      --indigo: #6366f1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px;
      background: radial-gradient(circle at top right, #111c33, var(--bg) 60%);
      color: var(--text);
      font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 22px;
    }}
    h1 {{ margin: 0; font-size: 2rem; letter-spacing: -0.02em; }}
    .sub {{ color: var(--muted); font-size: 0.95rem; margin-top: 4px; }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 0.85rem;
    }}
    .chip {{
      background: var(--card);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 999px;
      color: #cbd5e1;
    }}
    .chip strong {{ color: var(--accent); }}
    .table-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{
      text-align: left;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      background: #0b111e;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
    }}
    td {{
      padding: 14px 16px;
      border-bottom: 1px solid rgba(30, 41, 59, 0.7);
      vertical-align: middle;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .pos {{ font-weight: 800; font-size: 1.05rem; white-space: nowrap; }}
    .student-name {{ font-weight: 700; font-size: 1rem; }}
    .student-id {{ font-size: 0.78rem; color: var(--muted); font-family: monospace; }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.78rem;
      letter-spacing: 0.04em;
    }}
    .badge-icl {{ background: rgba(20, 184, 166, 0.16); color: #5eead4; border: 1px solid rgba(20, 184, 166, 0.4); }}
    .badge-lora {{ background: rgba(99, 102, 241, 0.18); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.45); }}
    .pct-main {{ font-weight: 700; font-size: 1rem; }}
    .pills {{ display: flex; gap: 6px; margin-top: 4px; flex-wrap: wrap; }}
    .pill {{
      font-size: 0.72rem;
      padding: 2px 7px;
      border-radius: 6px;
      font-family: monospace;
    }}
    .pill-ok {{ background: rgba(34, 197, 94, 0.15); color: #86efac; }}
    .pill-part {{ background: rgba(245, 158, 11, 0.15); color: #fcd34d; }}
    .time-main {{ font-weight: 600; }}
    .time-sub {{ font-size: 0.76rem; color: var(--muted); }}
    .lat {{ font-family: monospace; color: #cbd5e1; }}
    .score {{ font-weight: 800; font-size: 1.15rem; color: var(--amber); font-family: monospace; }}
    .empty {{ text-align: center; padding: 48px 20px; color: var(--muted); }}
    footer {{ margin-top: 14px; font-size: 0.8rem; color: var(--muted); text-align: right; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>🏁 UIS Grand Prix Foundationals Models</h1>
        <div class="sub">Tabla de posiciones oficial en vivo · Evaluación en pistas ocultas B1, B2 y B3</div>
      </div>
      <div class="meta">
        <span class="chip">Modelo base: <strong>{html.escape(MODEL_ID)}</strong></span>
        <span class="chip">Penalización latencia: <strong>{LATENCY_PENALTY_WEIGHT}×</strong></span>
        <span class="chip">Equipos: <strong>{len(filas)}</strong></span>
        <span class="chip">Auto-refresco: <strong>5s</strong></span>
      </div>
    </header>
    <div class="table-card">
      <table>
        <thead>
          <tr>
            <th>Posición</th>
            <th>Estudiante</th>
            <th>Estrategia</th>
            <th>% Promedio Pistas B</th>
            <th>Tiempo Total</th>
            <th>Latencia Promedio ms</th>
            <th>Score Final</th>
          </tr>
        </thead>
        <tbody>
          {cuerpo_tabla}
        </tbody>
      </table>
    </div>
    <footer>Fórmula: Score = (% Promedio × 10) + (Pistas Completadas × 150) − (Tiempo Físico + {LATENCY_PENALTY_WEIGHT} × Latencia Total) × 2.5</footer>
  </div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
