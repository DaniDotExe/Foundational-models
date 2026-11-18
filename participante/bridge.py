"""Cliente HTTP hacia el servidor central (RTX 3060 Ti local o Colab + trycloudflare).

Maneja timeouts, reintentos automáticos y medición exacta de latencia (rtt_ms)
para la evaluación oficial penalizada.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class ServerBridge:
    """Puente de comunicación síncrona con el backend FastAPI."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        student_id: Optional[str] = None,
        student_name: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        max_retries: int = 2,
    ) -> None:
        self.server_url = (
            server_url or os.getenv("SERVER_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.student_name = (
            student_name or os.getenv("STUDENT_NAME", "Oscar_Prueba")
        ).strip() or "Oscar_Prueba"
        id_base = student_id or os.getenv("STUDENT_ID", "").strip() or self.student_name
        self.student_id = re.sub(r"[^a-zA-Z0-9_-]", "_", id_base).lower() or "participante"
        self.timeout_sec = float(
            timeout_sec
            if timeout_sec is not None
            else os.getenv("REQUEST_TIMEOUT_SEC", "5.0")
        )
        self.max_retries = max(0, int(max_retries))
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def verificar_salud(self) -> Dict[str, Any]:
        """Consulta GET /health para verificar conexión, GPU y modelo cargado."""
        resp = self.session.get(
            f"{self.server_url}/health",
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        return resp.json()

    def entrenar_lora(
        self,
        jugadas: List[Dict[str, Any]],
        config_lora: Dict[str, Any],
        config_datos: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Envía las jugadas grabadas a POST /entrenar_lora."""
        payload = {
            "student_id": self.student_id,
            "jugadas": jugadas,
            "lora_r": int(config_lora.get("lora_r", 8)),
            "lora_alpha": int(config_lora.get("lora_alpha", 16)),
            "target_modules": list(
                config_lora.get("target_modules", ["q_proj", "v_proj"])
            ),
            "epochs": int(config_lora.get("epochs", 5)),
            "learning_rate": float(config_lora.get("learning_rate", 5e-4)),
            "num_ejemplos": int(config_lora.get("num_ejemplos", 150)),
            "precision_sensores": int(config_datos.get("precision_sensores", 50)),
            "ventana_historia": int(config_datos.get("ventana_historia", 3)),
        }
        train_timeout = max(600.0, self.timeout_sec * 60.0)
        resp = self.session.post(
            f"{self.server_url}/entrenar_lora",
            json=payload,
            timeout=train_timeout,
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Error del servidor ({resp.status_code}): {detail}")
        return resp.json()

    def decidir_accion(
        self,
        estrategia: str,
        estado_actual: Sequence[float],
        historial: Sequence[int],
        jugadas_referencia: Optional[List[Dict[str, Any]]],
        config_icl: Dict[str, Any],
        config_datos: Dict[str, Any],
        velocidad: Optional[float] = None,
        cp_dist: Optional[float] = None,
        cp_ang: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Consulta POST /decidir_accion con reintentos y mide RTT real en milisegundos."""
        est = estrategia.upper().strip()
        payload: Dict[str, Any] = {
            "student_id": self.student_id,
            "estrategia": est,
            "estado_actual": [float(x) for x in estado_actual[:5]],
            "historial": [int(a) for a in historial],
            "jugadas_referencia": jugadas_referencia if est == "ICL" else None,
            "k_shots": int(config_icl.get("k_shots", 10)),
            "seleccion": str(config_icl.get("seleccion", "knn")),
            "pesos_sensores": list(
                config_icl.get("pesos_sensores", [1.0, 1.5, 2.5, 1.5, 1.0])
            ),
            "temperature": float(config_icl.get("temperature", 0.0)),
            "top_p": float(config_icl.get("top_p", 0.9)),
            "precision_sensores": int(config_datos.get("precision_sensores", 50)),
            "ventana_historia": int(config_datos.get("ventana_historia", 3)),
        }
        if velocidad is not None:
            payload["velocidad"] = float(velocidad)
        if cp_dist is not None:
            payload["cp_dist"] = float(cp_dist)
        if cp_ang is not None:
            payload["cp_ang"] = float(cp_ang)

        t0 = time.perf_counter()
        last_err: Optional[Exception] = None

        for intento in range(self.max_retries + 1):
            try:
                resp = self.session.post(
                    f"{self.server_url}/decidir_accion",
                    json=payload,
                    timeout=self.timeout_sec,
                )
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    # Errores 4xx de configuración no se reintentan
                    if 400 <= resp.status_code < 500:
                        rtt_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                        return {
                            "accion": self._fallback_accion(estado_actual),
                            "latencia_gpu_ms": 0.0,
                            "rtt_ms": rtt_ms,
                            "ok": False,
                            "error": f"HTTP {resp.status_code}: {detail}",
                        }
                    raise RuntimeError(f"HTTP {resp.status_code}: {detail}")

                data = resp.json()
                rtt_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                accion = int(data.get("accion", 0))
                if accion not in (0, 1, 2, 3):
                    accion = self._fallback_accion(estado_actual)
                return {
                    "accion": accion,
                    "latencia_gpu_ms": float(data.get("latencia_gpu_ms", 0.0)),
                    "rtt_ms": rtt_ms,
                    "ok": True,
                    "error": None,
                }
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                last_err = exc
                if intento < self.max_retries:
                    time.sleep(0.15 * (intento + 1))

        rtt_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "accion": self._fallback_accion(estado_actual),
            "latencia_gpu_ms": 0.0,
            "rtt_ms": rtt_ms,
            "ok": False,
            "error": str(last_err) if last_err else "Error de conexión",
        }

    def registrar_puntaje(
        self,
        estrategia: str,
        resultados_pistas: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Envía los resultados oficiales de B1, B2, B3 a POST /registrar_puntaje."""
        payload = {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "estrategia": estrategia.upper().strip(),
            "resultados_pistas": resultados_pistas,
        }
        resp = self.session.post(
            f"{self.server_url}/registrar_puntaje",
            json=payload,
            timeout=max(10.0, self.timeout_sec),
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Error registrando puntaje ({resp.status_code}): {detail}")
        return resp.json()

    @staticmethod
    def _fallback_accion(estado_actual: Sequence[float]) -> int:
        frente = float(estado_actual[2]) if len(estado_actual) >= 3 else 1.0
        return 3 if frente < 0.25 else 0
