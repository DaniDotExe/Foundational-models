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

import base64
import html
import json
import logging
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
TRAYECTORIAS_PATH = BASE_DIR / "trayectorias_b.json"
_train_jobs_lock = threading.Lock()
_train_jobs: Dict[str, Dict[str, Any]] = {}
_student_examples_lock = threading.Lock()
_student_trained_examples: Dict[str, int] = {}
logger = logging.getLogger("grandprix_server")

F1_PUNTOS = [25.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0]
PESO_CARRERA = 0.55
PESO_VELOCIDAD = 0.20
PESO_DATOS = 0.15
PESO_LIMPIEZA = 0.10
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
    adapter_name: Optional[str] = None


class DecidirAccionRequest(BaseModel):
    student_id: str
    estrategia: str = Field(..., pattern="^(ICL|LORA|LORA_ICL|icl|lora|lora_icl)$")
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
    adapter_name: Optional[str] = None


class ResultadoPistaB(BaseModel):
    pista: str
    progreso_pct: float = Field(..., ge=0.0, le=100.0)
    tiempo_fisico_s: float = Field(..., ge=0.0)
    latencia_acumulada_s: float = Field(..., ge=0.0)
    latencia_promedio_ms: float = Field(default=0.0, ge=0.0)
    completada: bool = False
    choques: int = Field(default=0, ge=0)
    tiempo_fuera_s: float = Field(default=0.0, ge=0.0)
    trayectoria: Optional[List[List[float]]] = None


class RegistrarPuntajeRequest(BaseModel):
    student_id: str
    student_name: str
    estrategia: str
    resultados_pistas: Union[Dict[str, ResultadoPistaB], List[ResultadoPistaB]]
    num_ejemplos: Optional[int] = Field(default=None, ge=0)


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


def _cargar_trayectorias() -> Dict[str, Dict[str, Any]]:
    if not TRAYECTORIAS_PATH.exists():
        return {}
    try:
        data = json.loads(TRAYECTORIAS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _guardar_trayectorias(data: Dict[str, Dict[str, Any]]) -> None:
    TRAYECTORIAS_PATH.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def _puntos_f1(pos_idx_0: int, progreso_pct: float) -> float:
    if progreso_pct <= 0.0:
        return 0.0
    if pos_idx_0 < len(F1_PUNTOS):
        return F1_PUNTOS[pos_idx_0]
    return 0.5


def _calcular_rankings_f1(
    data: Dict[str, Dict[str, Any]],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    rows = [dict(v) for v in data.values()]
    for r in rows:
        n_ej = max(1, int(r.get("num_ejemplos") or 200))
        r["num_ejemplos"] = n_ej
        r["eficiencia_datos"] = round(float(r.get("progreso_promedio_pct", 0.0)) / n_ej, 3)

    top_carrera = sorted(
        rows,
        key=lambda r: (
            -int(r.get("pistas_completadas", 0)),
            -float(r.get("progreso_promedio_pct", 0.0)),
            float(r.get("tiempo_fisico_total_s", 1e9)),
        ),
    )
    for i, r in enumerate(top_carrera):
        r["pos_carrera"] = i + 1
        r["pts_carrera"] = _puntos_f1(i, float(r.get("progreso_promedio_pct", 0.0)))

    top_velocidad = sorted(
        rows,
        key=lambda r: (
            float(r.get("latencia_promedio_ms", 1e9)),
            -float(r.get("progreso_promedio_pct", 0.0)),
        ),
    )
    for i, r in enumerate(top_velocidad):
        r["pos_velocidad"] = i + 1
        r["pts_velocidad"] = _puntos_f1(i, float(r.get("progreso_promedio_pct", 0.0)))

    top_datos = sorted(
        rows,
        key=lambda r: (
            0 if float(r.get("progreso_promedio_pct", 0.0)) >= 20.0 else 1,
            -float(r.get("eficiencia_datos", 0.0)),
            -float(r.get("progreso_promedio_pct", 0.0)),
        ),
    )
    for i, r in enumerate(top_datos):
        r["pos_datos"] = i + 1
        prog = float(r.get("progreso_promedio_pct", 0.0))
        r["pts_datos"] = _puntos_f1(i, prog) if prog >= 20.0 else 0.0

    top_limpieza = sorted(
        rows,
        key=lambda r: (
            int(r.get("choques_totales", 999)),
            -float(r.get("progreso_promedio_pct", 0.0)),
            float(r.get("tiempo_fisico_total_s", 1e9)),
        ),
    )
    for i, r in enumerate(top_limpieza):
        r["pos_limpieza"] = i + 1
        r["pts_limpieza"] = _puntos_f1(i, float(r.get("progreso_promedio_pct", 0.0)))

    for r in rows:
        r["puntos_f1"] = round(
            PESO_CARRERA * r["pts_carrera"]
            + PESO_VELOCIDAD * r["pts_velocidad"]
            + PESO_DATOS * r["pts_datos"]
            + PESO_LIMPIEZA * r["pts_limpieza"],
            2,
        )

    general = sorted(
        rows,
        key=lambda r: (
            -float(r.get("puntos_f1", 0.0)),
            -float(r.get("pts_carrera", 0.0)),
            -float(r.get("progreso_promedio_pct", 0.0)),
            float(r.get("tiempo_fisico_total_s", 1e9)),
        ),
    )
    return general, top_carrera, top_velocidad, top_datos, top_limpieza


def _ordenar_ranking(data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    general, _, _, _, _ = _calcular_rankings_f1(data)
    return general


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


@app.get("/adaptadores/{student_id}")
def listar_adaptadores_estudiante(student_id: str) -> Dict[str, Any]:
    """Lista los adaptadores LoRA guardados (máx. 3 por participante)."""
    try:
        sid = sanitizar_student_id(student_id)
        return {"student_id": sid, "adaptadores": motor_global.listar_adaptadores(sid)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/adaptadores/{student_id}/{adapter_name}")
def eliminar_adaptador_estudiante(student_id: str, adapter_name: str) -> Dict[str, Any]:
    """Elimina un adaptador LoRA específico del estudiante."""
    try:
        sid = sanitizar_student_id(student_id)
        ok = motor_global.eliminar_adaptador(sid, adapter_name)
        return {"student_id": sid, "adapter_name": adapter_name, "eliminado": ok}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/adaptadores/{student_id}/{adapter_name}/exportar")
def exportar_adaptador_estudiante(student_id: str, adapter_name: str) -> Dict[str, Any]:
    """Exporta los archivos del adaptador LoRA en base64 para empaquetar en un solo .zip."""
    try:
        sid = sanitizar_student_id(student_id)
        akey = motor_global._adapter_key(sid, adapter_name)
        ruta = motor_global._resolver_dir_carga_peft(motor_global._ruta_adaptador(akey), akey)
        if not (ruta / "adapter_config.json").exists():
            raise HTTPException(status_code=404, detail=f"No existe el adaptador '{adapter_name}' para '{sid}'.")
        archivos: Dict[str, str] = {}
        for p in ruta.iterdir():
            if p.is_file():
                archivos[p.name] = base64.b64encode(p.read_bytes()).decode("ascii")
        return {"student_id": sid, "adapter_name": adapter_name, "files": archivos}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/admin/limpiar")
def limpiar_clase(clave: str = "") -> Dict[str, Any]:
    """Limpia ranking, trayectorias y adaptadores de prueba antes de iniciar la clase."""
    if clave != "dani":
        raise HTTPException(status_code=403, detail="Clave docente incorrecta.")
    with ranking_lock:
        if RANKING_PATH.exists():
            RANKING_PATH.unlink()
        if TRAYECTORIAS_PATH.exists():
            TRAYECTORIAS_PATH.unlink()
    eliminados = []
    for sid_or_dir in list(motor_global.adapters_dir.iterdir()):
        if sid_or_dir.is_dir() and not sid_or_dir.name.startswith("."):
            if "__" in sid_or_dir.name:
                sid_p, aname_p = sid_or_dir.name.split("__", 1)
                motor_global.eliminar_adaptador(sid_p, aname_p)
            else:
                motor_global.eliminar_adaptador(sid_or_dir.name, "default")
            eliminados.append(sid_or_dir.name)
    return {"status": "limpio", "adaptadores_eliminados": eliminados}


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
            adapter_name=req.adapter_name,
        )
        sid_clean = sanitizar_student_id(req.student_id)
        with _student_examples_lock:
            _student_trained_examples[sid_clean] = int(resultado.get("ejemplos_entrenados", len(jugadas_dict)))
        return resultado
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error entrenando LoRA: {exc}") from exc


def _ejecutar_job_lora(sid: str, req: EntrenarLoraRequest) -> None:
    def _on_prog(ep: int, tot: int, loss_ep: float) -> None:
        with _train_jobs_lock:
            job = _train_jobs.get(sid)
            if job:
                job["epoch"] = ep
                job["total_epochs"] = tot
                job["pct"] = round(min(99.0, (ep / max(1, tot)) * 100.0), 1)
                job["loss"] = loss_ep

    def _is_cancelled() -> bool:
        with _train_jobs_lock:
            job = _train_jobs.get(sid)
            return bool(job and job.get("cancel"))

    try:
        logger.info(
            "[ENTRENANDO] %s | ejemplos=%d | epochs=%d | r=%d",
            sid, len(req.jugadas), req.epochs, req.lora_r,
        )
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
            adapter_name=req.adapter_name,
            progress_cb=_on_prog,
            cancel_check=_is_cancelled,
        )
        with _student_examples_lock:
            _student_trained_examples[sid] = int(resultado.get("ejemplos_entrenados", len(jugadas_dict)))
        logger.info(
            "[LORA LISTO] %s | loss=%.4f | tiempo=%.1fs",
            sid, resultado.get("loss_final", 0.0), resultado.get("tiempo_entrenamiento_s", 0.0),
        )
        with _train_jobs_lock:
            _train_jobs[sid] = {
                "status": "done",
                "pct": 100.0,
                "epoch": req.epochs,
                "total_epochs": req.epochs,
                "loss": resultado.get("loss_final"),
                "result": resultado,
                "error": None,
            }
    except InterruptedError as exc:
        logger.info("[LORA CANCELADO] %s", sid)
        with _train_jobs_lock:
            _train_jobs[sid] = {"status": "cancelled", "result": None, "error": str(exc)}
    except Exception as exc:
        logger.error("[LORA ERROR] %s | %s", sid, exc)
        with _train_jobs_lock:
            _train_jobs[sid] = {"status": "error", "result": None, "error": str(exc)}


@app.post("/entrenar_lora_async")
def entrenar_lora_async(req: EntrenarLoraRequest) -> Dict[str, Any]:
    """Inicia el entrenamiento LoRA en segundo plano para evitar timeouts HTTP (Error 524 de Cloudflare)."""
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
        sid = sanitizar_student_id(req.student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _train_jobs_lock:
        actual = _train_jobs.get(sid)
        if actual and actual.get("status") == "running":
            return {"status": "running", "student_id": sid}
        _train_jobs[sid] = {
            "status": "running",
            "pct": 0.0,
            "epoch": 0,
            "total_epochs": req.epochs,
            "loss": None,
            "cancel": False,
            "result": None,
            "error": None,
        }

    t = threading.Thread(target=_ejecutar_job_lora, args=(sid, req), daemon=True)
    t.start()
    return {"status": "running", "student_id": sid}


@app.post("/entrenar_lora_cancelar/{student_id}")
def entrenar_lora_cancelar(student_id: str) -> Dict[str, Any]:
    """Solicita cancelar el entrenamiento LoRA en curso de `student_id`."""
    try:
        sid = sanitizar_student_id(student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _train_jobs_lock:
        job = _train_jobs.get(sid)
        if job and job.get("status") == "running":
            job["cancel"] = True
            return {"status": "cancelling", "student_id": sid}
    return {"status": "idle", "student_id": sid}


@app.get("/entrenar_lora_estado/{student_id}")
def entrenar_lora_estado(student_id: str) -> Dict[str, Any]:
    """Consulta el estado del entrenamiento LoRA asíncrono."""
    try:
        sid = sanitizar_student_id(student_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _train_jobs_lock:
        job = _train_jobs.get(sid)
    if not job:
        raise HTTPException(status_code=404, detail=f"No hay entrenamiento activo para '{sid}'.")
    return job


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
            adapter_name=req.adapter_name,
        )
        logger.info(
            "[SIMULANDO] %s | %s | acc=%s | gpu=%.1fms",
            req.student_id, req.estrategia.upper(), resultado["accion"], resultado["latencia_gpu_ms"],
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

    est_limpia = req.estrategia.upper().strip()
    if req.num_ejemplos is not None and req.num_ejemplos > 0:
        num_ejemplos_usados = int(req.num_ejemplos)
    else:
        with _student_examples_lock:
            num_ejemplos_usados = int(_student_trained_examples.get(sid, 200))

    entrada = {
        "entry_id": f"{sid}__{est_limpia}",
        "student_id": sid,
        "student_name": req.student_name.strip() or sid,
        "estrategia": est_limpia,
        "num_ejemplos": num_ejemplos_usados,
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

    entry_key = f"{sid}__{est_limpia}"
    with ranking_lock:
        ranking = _cargar_ranking()
        ranking[entry_key] = entrada
        _guardar_ranking(ranking)
        ordenado = _ordenar_ranking(ranking)

        tray_all = _cargar_trayectorias()
        tray_pistas = {
            p: pistas_map[p].trayectoria
            for p in requeridas
            if pistas_map[p].trayectoria
        }
        if tray_pistas:
            tray_all[entry_key] = {
                "entry_id": entry_key,
                "student_id": sid,
                "student_name": entrada["student_name"],
                "estrategia": est_limpia,
                "pistas": tray_pistas,
            }
            _guardar_trayectorias(tray_all)

    reg_final = next((row for row in ordenado if row.get("entry_id", f"{row['student_id']}__{row['estrategia']}") == entry_key), entrada)
    posicion = next(
        (idx + 1 for idx, row in enumerate(ordenado) if row.get("entry_id", f"{row['student_id']}__{row['estrategia']}") == entry_key),
        len(ordenado),
    )
    return {
        "posicion": posicion,
        "total_estudiantes": len(ordenado),
        "registro": reg_final,
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


@app.get("/trayectorias_b")
def obtener_trayectorias_b() -> Dict[str, Any]:
    """Devuelve las trayectorias frame a frame de B1, B2 y B3 para el visor multi-carro."""
    with ranking_lock:
        tray_all = _cargar_trayectorias()
    por_pista: Dict[str, List[Dict[str, Any]]] = {"B1": [], "B2": [], "B3": []}
    for entry in tray_all.values():
        pistas = entry.get("pistas", {})
        for p in ("B1", "B2", "B3"):
            if p in pistas and pistas[p]:
                por_pista[p].append({
                    "entry_id": entry.get("entry_id"),
                    "student_name": entry.get("student_name"),
                    "estrategia": entry.get("estrategia"),
                    "trayectoria": pistas[p],
                })
    return {"pistas": por_pista}


@app.get("/ranking", response_class=HTMLResponse)
def ranking_html() -> str:
    """Dashboard HTML en vivo con Campeonato General F1 y 4 sub-rankings explicados."""
    with ranking_lock:
        filas, top_carrera, top_velocidad, top_datos, top_limpieza = _calcular_rankings_f1(_cargar_ranking())

    def _badge_cls(est: str) -> str:
        if est == "LORA":
            return "badge-lora"
        if est == "LORA_ICL":
            return "badge-both"
        return "badge-icl"

    filas_html = []
    for idx, row in enumerate(filas, start=1):
        nombre = html.escape(str(row.get("student_name", row.get("student_id", "-"))))
        sid = html.escape(str(row.get("student_id", "")))
        est = html.escape(str(row.get("estrategia", "-")))
        badge_cls = _badge_cls(est)

        pistas_b = row.get("pistas_b", {})
        pills = []
        for p in ("B1", "B2", "B3"):
            info = pistas_b.get(p, {})
            pct = float(info.get("progreso_pct", 0.0))
            ok_cls = "pill-ok" if info.get("completada") else "pill-part"
            pills.append(f'<span class="pill {ok_cls}">{p}: {pct:.0f}%</span>')

        medalla = {1: "🥇 ", 2: "🥈 ", 3: "🥉 "}.get(idx, "")
        progreso_prom = float(row.get("progreso_promedio_pct", 0.0))
        t_fis = float(row.get("tiempo_fisico_total_s", 0.0))
        lat_ms = float(row.get("latencia_promedio_ms", 0.0))
        pts_f1 = float(row.get("puntos_f1", 0.0))

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
                <div class="pct-main">{progreso_prom:.1f}% <span class="time-sub">({t_fis:.1f}s)</span></div>
                <div class="pills">{''.join(pills)}</div>
              </td>
              <td class="sub-pts">#{row['pos_carrera']} <span class="pts-tag">({row['pts_carrera']:.1f})</span></td>
              <td class="sub-pts">#{row['pos_velocidad']} <span class="pts-tag">({row['pts_velocidad']:.1f})</span><div class="time-sub">{lat_ms:.0f} ms</div></td>
              <td class="sub-pts">#{row['pos_datos']} <span class="pts-tag">({row['pts_datos']:.1f})</span><div class="time-sub">{row['num_ejemplos']} ej</div></td>
              <td class="sub-pts">#{row['pos_limpieza']} <span class="pts-tag">({row['pts_limpieza']:.1f})</span><div class="time-sub">{row['choques_totales']} choq</div></td>
              <td class="score">{pts_f1:.2f}</td>
            </tr>
            """
        )

    cuerpo_tabla = (
        "\n".join(filas_html)
        if filas_html
        else '<tr><td colspan="9" class="empty">Aún no hay evaluaciones oficiales registradas en Pistas B. Ejecuta el modo [4] en el cliente.</td></tr>'
    )

    def _render_subtabla(lista: List[Dict[str, Any]], metrica_fn: Any, pts_key: str) -> str:
        if not lista:
            return '<tr><td colspan="4" class="empty">Sin datos aún.</td></tr>'
        out = []
        for i, r in enumerate(lista, start=1):
            nm = html.escape(str(r.get("student_name", r.get("student_id", "-"))))
            es = html.escape(str(r.get("estrategia", "-")))
            out.append(
                f"<tr><td class='pos'>#{i}</td><td><strong>{nm}</strong> <span class='badge {_badge_cls(es)}'>{es}</span></td>"
                f"<td class='lat'>{metrica_fn(r)}</td><td class='score'>{r[pts_key]:.1f} pts</td></tr>"
            )
        return "\n".join(out)

    sub_carrera = _render_subtabla(
        top_carrera,
        lambda r: f"{r['pistas_completadas']}/3 | {r['progreso_promedio_pct']:.1f}% | {r['tiempo_fisico_total_s']:.1f}s",
        "pts_carrera",
    )
    sub_velocidad = _render_subtabla(
        top_velocidad,
        lambda r: f"{r['latencia_promedio_ms']:.1f} ms/frame",
        "pts_velocidad",
    )
    sub_datos = _render_subtabla(
        top_datos,
        lambda r: f"{r['eficiencia_datos']:.2f} %/ej ({r['num_ejemplos']} ej)",
        "pts_datos",
    )
    sub_limpieza = _render_subtabla(
        top_limpieza,
        lambda r: f"{r['choques_totales']} choques | {r['progreso_promedio_pct']:.1f}%",
        "pts_limpieza",
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="refresh" content="5" />
  <title>UIS Grand Prix Foundationals Models — Ranking F1 en Vivo</title>
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
      padding: 24px;
      background: radial-gradient(circle at top right, #111c33, var(--bg) 60%);
      color: var(--text);
      font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif;
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0; font-size: 1.9rem; letter-spacing: -0.02em; }}
    h2 {{ margin: 22px 0 10px 0; font-size: 1.15rem; color: var(--text); }}
    .sub {{ color: var(--muted); font-size: 0.92rem; margin-top: 4px; }}
    .meta {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 0.84rem; }}
    .chip {{ background: var(--card); border: 1px solid var(--border); padding: 6px 12px; border-radius: 999px; color: #cbd5e1; }}
    .chip strong {{ color: var(--accent); }}
    .explain-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .explain-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    .explain-card h3 {{ margin: 0 0 6px 0; font-size: 0.98rem; color: var(--accent); }}
    .explain-card p {{ margin: 0; font-size: 0.82rem; color: var(--muted); line-height: 1.45; }}
    .table-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.45);
    }}
    .sub-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{
      text-align: left;
      font-size: 0.74rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      background: #0b111e;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(30, 41, 59, 0.7);
      vertical-align: middle;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}
    .pos {{ font-weight: 800; font-size: 1rem; white-space: nowrap; }}
    .student-name {{ font-weight: 700; font-size: 0.96rem; }}
    .student-id {{ font-size: 0.76rem; color: var(--muted); font-family: monospace; }}
    .badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.74rem;
      letter-spacing: 0.04em;
    }}
    .badge-icl {{ background: rgba(20, 184, 166, 0.16); color: #5eead4; border: 1px solid rgba(20, 184, 166, 0.4); }}
    .badge-lora {{ background: rgba(99, 102, 241, 0.18); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.45); }}
    .badge-both {{ background: rgba(245, 158, 11, 0.18); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.45); }}
    .pct-main {{ font-weight: 700; font-size: 0.95rem; }}
    .pills {{ display: flex; gap: 5px; margin-top: 4px; flex-wrap: wrap; }}
    .pill {{ font-size: 0.7rem; padding: 2px 6px; border-radius: 5px; font-family: monospace; }}
    .pill-ok {{ background: rgba(34, 197, 94, 0.15); color: #86efac; }}
    .pill-part {{ background: rgba(245, 158, 11, 0.15); color: #fcd34d; }}
    .sub-pts {{ font-weight: 700; font-size: 0.9rem; }}
    .pts-tag {{ color: var(--accent); font-weight: 600; font-size: 0.82rem; }}
    .time-sub {{ font-size: 0.74rem; color: var(--muted); font-weight: 400; }}
    .lat {{ font-family: monospace; color: #cbd5e1; font-size: 0.85rem; }}
    .score {{ font-weight: 800; font-size: 1.08rem; color: var(--amber); font-family: monospace; }}
    .tab-bar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 14px 0 16px 0;
    }}
    .tab-btn {{
      background: var(--card);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 10px 16px;
      border-radius: 10px;
      font-weight: 700;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .tab-btn:hover {{ color: var(--text); border-color: var(--accent); }}
    .tab-btn.active {{
      background: rgba(20, 184, 166, 0.16);
      color: #5eead4;
      border-color: var(--accent);
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .panel-desc {{
      background: rgba(16, 23, 38, 0.85);
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      padding: 12px 16px;
      border-radius: 10px;
      margin-bottom: 12px;
      font-size: 0.88rem;
      color: #cbd5e1;
    }}
    .empty {{ text-align: center; padding: 36px 20px; color: var(--muted); }}
    footer {{ margin-top: 16px; font-size: 0.82rem; color: var(--muted); text-align: right; }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>🏁 UIS Grand Prix — Campeonato F1 en Vivo</h1>
        <div class="sub">Evaluación oficial en pistas ocultas B1, B2 y B3 · Escala F1 por categoría: 25, 18, 15, 12, 10, 8, 6, 4, 2, 1 pts</div>
      </div>
      <div class="meta">
        <span class="chip">Modelo base: <strong>{html.escape(MODEL_ID)}</strong></span>
        <span class="chip">Participaciones: <strong>{len(filas)}</strong></span>
        <span class="chip">Auto-refresco: <strong>5s</strong></span>
      </div>
    </header>

    <div class="tab-bar">
      <button class="tab-btn active" data-tab="general" onclick="setTab('general')">🏆 Campeonato General F1</button>
      <button class="tab-btn" data-tab="carrera" onclick="setTab('carrera')">🏁 Carrera (55%)</button>
      <button class="tab-btn" data-tab="velocidad" onclick="setTab('velocidad')">⚡ Velocidad (20%)</button>
      <button class="tab-btn" data-tab="datos" onclick="setTab('datos')">🧬 Few-Shot (15%)</button>
      <button class="tab-btn" data-tab="limpieza" onclick="setTab('limpieza')">🛡️ Limpieza (10%)</button>
    </div>

    <div id="tab-general" class="tab-panel active">
      <div class="panel-desc">
        <strong>🏆 Campeonato General (Final):</strong> Promedio ponderado <code>0.55·Carrera + 0.20·Velocidad + 0.15·Few-Shot + 0.10·Limpieza</code>. Cada categoría reparte puntos F1 (25, 18, 15, 12, 10, 8, 6, 4, 2, 1).
      </div>
      <div class="table-card">
        <table>
          <thead>
            <tr>
              <th>Pos</th>
              <th>Estudiante</th>
              <th>Estrategia</th>
              <th>Pistas B</th>
              <th>🏁 Carrera (55%)</th>
              <th>⚡ Velocidad (20%)</th>
              <th>🧬 Few-Shot (15%)</th>
              <th>🛡️ Limpieza (10%)</th>
              <th>Pts F1</th>
            </tr>
          </thead>
          <tbody>
            {cuerpo_tabla}
          </tbody>
        </table>
      </div>
    </div>

    <div id="tab-carrera" class="tab-panel">
      <div class="panel-desc">
        <strong>🏁 Ranking de Carrera (Peso 55%):</strong> Ordena por pistas B completadas (B1, B2, B3), luego mayor % de progreso promedio y menor tiempo físico total.
      </div>
      <div class="table-card">
        <table>
          <thead><tr><th>Pos</th><th>Estudiante</th><th>Completadas · % · Tiempo</th><th>Puntos F1</th></tr></thead>
          <tbody>{sub_carrera}</tbody>
        </table>
      </div>
    </div>

    <div id="tab-velocidad" class="tab-panel">
      <div class="panel-desc">
        <strong>⚡ Ranking de Velocidad de Inferencia (Peso 20%):</strong> Ordena por menor latencia promedio (ms/frame) en GPU. Premia modelos compactos y prompts cortos (LoRA zero-shot).
      </div>
      <div class="table-card">
        <table>
          <thead><tr><th>Pos</th><th>Estudiante</th><th>Latencia Promedio</th><th>Puntos F1</th></tr></thead>
          <tbody>{sub_velocidad}</tbody>
        </table>
      </div>
    </div>

    <div id="tab-datos" class="tab-panel">
      <div class="panel-desc">
        <strong>🧬 Ranking Eficiencia Few-Shot (Peso 15%):</strong> Ordena por mayor <code>% progreso / #ejemplos</code>. Requiere al menos 20% de progreso promedio para puntuar.
      </div>
      <div class="table-card">
        <table>
          <thead><tr><th>Pos</th><th>Estudiante</th><th>Eficiencia (% / ejemplo)</th><th>Puntos F1</th></tr></thead>
          <tbody>{sub_datos}</tbody>
        </table>
      </div>
    </div>

    <div id="tab-limpieza" class="tab-panel">
      <div class="panel-desc">
        <strong>🛡️ Ranking de Conducción Limpia (Peso 10%):</strong> Ordena por menor número total de choques contra obstáculos en B1–B3 (desempata por mayor progreso %).
      </div>
      <div class="table-card">
        <table>
          <thead><tr><th>Pos</th><th>Estudiante</th><th>Choques · Progreso</th><th>Puntos F1</th></tr></thead>
          <tbody>{sub_limpieza}</tbody>
        </table>
      </div>
    </div>

    <footer>Para proyectar la carrera conjunta (Fantasma / Legendario), usa el botón <strong>[C] Ver Carrera Final (B1-B3)</strong> en el simulador.</footer>
  </div>
  <script>
    function setTab(name) {{
      localStorage.setItem('uis_gp_tab', name);
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
    }}
    const saved = localStorage.getItem('uis_gp_tab');
    if (saved && document.getElementById('tab-' + saved)) {{
      setTab(saved);
    }}
  </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
