"""Motor de inferencia ICL (KNN / Random) y entrenamiento PEFT/LoRA sobre Qwen2.5-1.5B-Instruct.

Mantiene una única instancia del modelo base en VRAM y protege todas las operaciones
críticas de GPU (entrenamiento LoRA, cambio de adaptadores y generación) mediante un
`threading.Lock()` global para atender hasta 12 estudiantes en paralelo sin mezclar pesos.
"""
from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("motor_qwen")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

# Lock global para aislar operaciones críticas en GPU entre hilos concurrentes
gpu_lock = threading.Lock()

ACCIONES_VALIDAS = {0, 1, 2, 3}
TARGET_MODULES_PERMITIDOS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
SYSTEM_HEADER = (
    "Piloto F1. Acciones: 0=recto, 1=izquierda, 2=derecha, 3=frenar. "
    "Responde solo 1 digito.\n"
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


MODEL_ID: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
TORCH_DTYPE_STR: str = os.getenv("TORCH_DTYPE", "bfloat16").lower()
DEVICE_CFG: str = os.getenv("DEVICE", "cuda").lower()
_raw_adapters_dir = os.getenv("ADAPTERS_DIR", "./adapters_cache")
ADAPTERS_DIR: Path = (
    Path(_raw_adapters_dir)
    if Path(_raw_adapters_dir).is_absolute()
    else (BASE_DIR / _raw_adapters_dir).resolve()
)
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

MAX_K_SHOTS: int = _env_int("MAX_K_SHOTS", 30)
MAX_LORA_EPOCHS: int = _env_int("MAX_LORA_EPOCHS", 15)
MAX_LORA_R: int = _env_int("MAX_LORA_R", 32)
LATENCY_PENALTY_WEIGHT: float = _env_float("LATENCY_PENALTY_WEIGHT", 1.5)


def sanitizar_student_id(student_id: str) -> str:
    limpio = re.sub(r"[^a-zA-Z0-9_-]", "_", student_id.strip())
    if not limpio:
        raise ValueError("student_id no puede estar vacío.")
    return limpio


def discretizar_sensores(sensores: Sequence[float], precision: int = 50) -> List[int]:
    """Convierte los 5 sensores normalizados [0..1] a enteros discretos en [0..precision]."""
    prec = max(5, min(500, int(precision)))
    vals = list(sensores[:5])
    while len(vals) < 5:
        vals.append(1.0)
    return [int(round(max(0.0, min(1.0, float(v))) * prec)) for v in vals]


def formatear_historial(historial: Sequence[int], ventana_historia: int = 3) -> str:
    ventana = max(1, min(8, int(ventana_historia)))
    ultimos = [int(a) for a in historial[-ventana:]] if historial else [0]
    return ",".join(str(a) for a in ultimos)


def formatear_estado(
    sensores: Sequence[float],
    historial: Sequence[int],
    precision: int = 50,
    ventana_historia: int = 3,
) -> str:
    """Formato compacto: S:<s1>,<s2>,<s3>,<s4>,<s5>|H:<historial>."""
    s_disc = discretizar_sensores(sensores, precision=precision)
    h_str = formatear_historial(historial, ventana_historia=ventana_historia)
    return f"S:{','.join(str(x) for x in s_disc)}|H:{h_str}"


def formatear_ejemplo(
    muestra: Dict[str, Any],
    precision: int = 50,
    ventana_historia: int = 3,
) -> str:
    """Formato compacto de entrenamiento/ICL: S:...|H:... -> <accion>."""
    estado_str = formatear_estado(
        muestra.get("sensores", [1.0] * 5),
        muestra.get("historial", [0]),
        precision=precision,
        ventana_historia=ventana_historia,
    )
    accion = int(muestra.get("accion", 0))
    if accion not in ACCIONES_VALIDAS:
        accion = 0
    return f"{estado_str} -> {accion}"


def seleccionar_ejemplos_icl(
    estado_actual: Sequence[float],
    jugadas_referencia: List[Dict[str, Any]],
    k_shots: int,
    seleccion: str = "knn",
    pesos_sensores: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """Selecciona `k_shots` jugadas usando KNN ponderado o muestreo aleatorio."""
    if not jugadas_referencia or k_shots <= 0:
        return []

    k = min(max(1, min(k_shots, MAX_K_SHOTS)), len(jugadas_referencia))
    modo = seleccion.lower().strip()

    if modo == "random":
        return random.sample(jugadas_referencia, k)

    # KNN ponderado sobre los 5 rayos de distancia normalizados
    if pesos_sensores is not None and len(pesos_sensores) == 5:
        pesos = np.array([max(0.0, float(w)) for w in pesos_sensores], dtype=np.float64)
    else:
        pesos = np.ones(5, dtype=np.float64)

    q = np.array([max(0.0, min(1.0, float(x))) for x in estado_actual[:5]], dtype=np.float64)
    distancias: List[tuple[float, int]] = []
    for idx, jugada in enumerate(jugadas_referencia):
        s_ref = jugada.get("sensores", [1.0] * 5)[:5]
        r = np.array([max(0.0, min(1.0, float(x))) for x in s_ref], dtype=np.float64)
        diff = q - r
        dist = float(np.sum(pesos * (diff * diff)))
        distancias.append((dist, idx))

    distancias.sort(key=lambda item: item[0])
    top_k = [jugadas_referencia[idx] for _, idx in distancias[:k]]
    # Ordenar de más lejano a más cercano para que el ejemplo más similar quede justo antes de la consulta
    top_k.reverse()
    return top_k


def parsear_accion(texto_generado: str, estado_actual: Sequence[float]) -> int:
    """Extrae el primer dígito válido {0,1,2,3} o aplica fallback seguro."""
    match = re.search(r"[0-3]", texto_generado)
    if match:
        return int(match.group(0))
    # Fallback seguro: si el obstáculo frontal está muy cerca, frenar suave (3); si no, recto (0)
    frente = float(estado_actual[2]) if len(estado_actual) >= 3 else 1.0
    return 3 if frente < 0.22 else 0


class MotorQwen:
    """Encapsula el modelo Qwen2.5-1.5B-Instruct, motor ICL y adaptadores PEFT/LoRA."""

    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self.adapters_dir = ADAPTERS_DIR
        self.mock_mode = os.getenv("MOCK_LLM", "0") == "1"

        self.tokenizer: Any = None
        self.base_model: Any = None
        self.peft_model: Any = None
        self.device: str = "cpu"
        self.dtype: Any = None
        self.loaded_adapters: set[str] = set()
        self._initialized: bool = False

    def inicializar(self) -> None:
        """Carga el modelo base una sola vez (thread-safe)."""
        if self._initialized:
            return
        with gpu_lock:
            if self._initialized:
                return
            self._cargar_modelo_bloqueado()
            self._initialized = True

    def _cargar_modelo_bloqueado(self) -> None:
        if self.mock_mode:
            logger.warning("MOCK_LLM=1 activo: usando motor heurístico liviano para pruebas sin descargar Qwen.")
            self.device = "cpu"
            self.dtype = "mock"
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_cuda = DEVICE_CFG == "cuda" and torch.cuda.is_available()
        if DEVICE_CFG == "cuda" and not use_cuda:
            logger.warning("CUDA solicitado en .env pero no disponible; usando CPU.")
        self.device = "cuda" if use_cuda else "cpu"

        if self.device == "cuda":
            if TORCH_DTYPE_STR == "bfloat16":
                if torch.cuda.is_bf16_supported():
                    self.dtype = torch.bfloat16
                else:
                    logger.warning("GPU sin soporte nativo bfloat16; usando float16.")
                    self.dtype = torch.float16
            elif TORCH_DTYPE_STR == "float16":
                self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = torch.float32

        logger.info("Cargando tokenizer y modelo base %s en %s (%s)...", self.model_id, self.device, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                attn_implementation="sdpa",
            )
        except (TypeError, ValueError) as exc:
            logger.warning("attn_implementation='sdpa' no soportado (%s); cargando implementación estándar.", exc)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
            )

        self.base_model.to(self.device)
        self.base_model.eval()
        logger.info("Modelo %s cargado correctamente en %s.", self.model_id, self.device)

    def info_salud(self) -> Dict[str, Any]:
        """Devuelve estado de GPU, VRAM libre y adaptadores cargados."""
        self.inicializar()
        gpu_activa = False
        gpu_nombre = "CPU / Mock"
        vram_libre_mb = 0.0
        vram_total_mb = 0.0

        if not self.mock_mode:
            import torch

            if torch.cuda.is_available() and self.device == "cuda":
                gpu_activa = True
                gpu_nombre = torch.cuda.get_device_name(0)
                libre, total = torch.cuda.mem_get_info(0)
                vram_libre_mb = round(libre / (1024 ** 2), 1)
                vram_total_mb = round(total / (1024 ** 2), 1)

        adaptadores_disco = [
            p.name for p in self.adapters_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]

        return {
            "status": "ok",
            "mock_mode": self.mock_mode,
            "model_id": self.model_id,
            "device": self.device,
            "torch_dtype": str(self.dtype),
            "gpu_activa": gpu_activa,
            "gpu_nombre": gpu_nombre,
            "vram_libre_mb": vram_libre_mb,
            "vram_total_mb": vram_total_mb,
            "adaptadores_en_memoria": sorted(self.loaded_adapters),
            "adaptadores_en_disco": sorted(adaptadores_disco),
            "limites": {
                "max_k_shots": MAX_K_SHOTS,
                "max_lora_epochs": MAX_LORA_EPOCHS,
                "max_lora_r": MAX_LORA_R,
                "latency_penalty_weight": LATENCY_PENALTY_WEIGHT,
            },
        }

    def _ruta_adaptador(self, student_id: str) -> Path:
        return self.adapters_dir / sanitizar_student_id(student_id)

    def _resolver_dir_carga_peft(self, student_dir: Path, sid: str) -> Path:
        """Localiza el directorio real que contiene adapter_config.json."""
        if (student_dir / "adapter_config.json").exists():
            return student_dir
        nested = student_dir / sid
        if (nested / "adapter_config.json").exists():
            return nested
        return student_dir

    def _asegurar_adaptador_cargado(self, sid: str) -> None:
        """Asume gpu_lock adquirido. Carga el adaptador del estudiante desde disco si no está en RAM."""
        from peft import PeftModel

        if self.peft_model is not None and sid in self.peft_model.peft_config:
            self.peft_model.set_adapter(sid)
            return

        student_dir = self._ruta_adaptador(sid)
        if not student_dir.exists():
            raise FileNotFoundError(
                f"El estudiante '{sid}' aún no tiene un adaptador LoRA entrenado. "
                "Ejecuta primero el modo [2] Entrenar Piloto."
            )

        load_dir = self._resolver_dir_carga_peft(student_dir, sid)
        if not (load_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No se encontró adapter_config.json para '{sid}' en {student_dir}."
            )

        if self.peft_model is None:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model,
                str(load_dir),
                adapter_name=sid,
                is_trainable=False,
            )
            self.peft_model.to(self.device)
        else:
            self.peft_model.load_adapter(str(load_dir), adapter_name=sid, is_trainable=False)

        self.peft_model.set_adapter(sid)
        self.peft_model.eval()
        self.loaded_adapters.add(sid)

    def entrenar_lora(
        self,
        student_id: str,
        jugadas: List[Dict[str, Any]],
        lora_r: int,
        lora_alpha: int,
        target_modules: List[str],
        epochs: int,
        learning_rate: float,
        num_ejemplos: int,
        precision_sensores: int = 50,
        ventana_historia: int = 3,
    ) -> Dict[str, Any]:
        """Entrena un adaptador LoRA aislado para `student_id` y lo guarda en disco."""
        self.inicializar()
        sid = sanitizar_student_id(student_id)

        if not (1 <= lora_r <= MAX_LORA_R):
            raise ValueError(f"lora_r={lora_r} excede el límite permitido [1, {MAX_LORA_R}].")
        if not (1 <= epochs <= MAX_LORA_EPOCHS):
            raise ValueError(f"epochs={epochs} excede el límite permitido [1, {MAX_LORA_EPOCHS}].")
        if not target_modules:
            raise ValueError("target_modules no puede estar vacío.")
        invalid_mods = set(target_modules) - TARGET_MODULES_PERMITIDOS
        if invalid_mods:
            raise ValueError(f"target_modules inválidos: {sorted(invalid_mods)}")
        if not jugadas:
            raise ValueError("La lista de jugadas está vacía; graba primero jugadas en las pistas A.")

        n_cap = max(1, min(int(num_ejemplos), len(jugadas)))
        if len(jugadas) > n_cap:
            indices = np.linspace(0, len(jugadas) - 1, num=n_cap, dtype=int)
            muestras = [jugadas[i] for i in indices]
        else:
            muestras = list(jugadas)

        if self.mock_mode:
            t0 = time.perf_counter()
            student_dir = self._ruta_adaptador(sid)
            student_dir.mkdir(parents=True, exist_ok=True)
            (student_dir / "mock_adapter.json").write_text(
                f'{{"student_id": "{sid}", "ejemplos": {len(muestras)}, "lora_r": {lora_r}}}',
                encoding="utf-8",
            )
            self.loaded_adapters.add(sid)
            time.sleep(0.05)
            return {
                "student_id": sid,
                "tiempo_entrenamiento_s": round(time.perf_counter() - t0, 3),
                "loss_final": 0.1842,
                "historial_loss": [0.95, 0.52, 0.1842],
                "ejemplos_entrenados": len(muestras),
                "adapter_dir": str(student_dir),
            }

        import torch
        from peft import LoraConfig, get_peft_model

        t0 = time.perf_counter()
        with gpu_lock:
            lora_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=list(target_modules),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            if self.peft_model is None:
                self.peft_model = get_peft_model(
                    self.base_model, lora_config, adapter_name=sid
                )
            else:
                if sid in self.peft_model.peft_config:
                    self.peft_model.delete_adapter(sid)
                self.peft_model.add_adapter(sid, lora_config)
                self.peft_model.set_adapter(sid)

            # Activar gradientes exclusivamente para el adaptador de este estudiante
            trainable_params = []
            for name, param in self.peft_model.named_parameters():
                es_del_alumno = f".{sid}." in name
                param.requires_grad = es_del_alumno
                if es_del_alumno:
                    trainable_params.append(param)

            if not trainable_params:
                raise RuntimeError(f"No se encontraron parámetros entrenables para el adaptador '{sid}'.")

            # Preparar pares tokenizados con máscara -100 sobre el prompt
            encoded_samples: List[tuple[List[int], List[int]]] = []
            for m in muestras:
                estado_str = formatear_estado(
                    m.get("sensores", [1.0] * 5),
                    m.get("historial", [0]),
                    precision=precision_sensores,
                    ventana_historia=ventana_historia,
                )
                accion = int(m.get("accion", 0))
                if accion not in ACCIONES_VALIDAS:
                    accion = 0
                prompt = f"{SYSTEM_HEADER}{estado_str} ->"
                full_text = f"{prompt} {accion}"

                prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
                full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids

                if len(full_ids) <= len(prompt_ids):
                    labels = [-100] * (len(full_ids) - 1) + [full_ids[-1]]
                else:
                    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
                encoded_samples.append((full_ids, labels))

            pad_id = int(self.tokenizer.pad_token_id)
            optimizer = torch.optim.AdamW(trainable_params, lr=float(learning_rate))
            batch_size = min(8, len(encoded_samples))

            self.peft_model.train()
            historial_loss: List[float] = []

            for _epoch in range(int(epochs)):
                random.shuffle(encoded_samples)
                epoch_losses: List[float] = []
                for start in range(0, len(encoded_samples), batch_size):
                    batch = encoded_samples[start:start + batch_size]
                    max_len = max(len(x[0]) for x in batch)
                    input_ids_b = []
                    attn_mask_b = []
                    labels_b = []

                    for ids, lbls in batch:
                        pad_n = max_len - len(ids)
                        input_ids_b.append([pad_id] * pad_n + ids)
                        attn_mask_b.append([0] * pad_n + [1] * len(ids))
                        labels_b.append([-100] * pad_n + lbls)

                    input_tensor = torch.tensor(input_ids_b, dtype=torch.long, device=self.device)
                    mask_tensor = torch.tensor(attn_mask_b, dtype=torch.long, device=self.device)
                    label_tensor = torch.tensor(labels_b, dtype=torch.long, device=self.device)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = self.peft_model(
                        input_ids=input_tensor,
                        attention_mask=mask_tensor,
                        labels=label_tensor,
                    )
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu().item()))

                historial_loss.append(round(float(np.mean(epoch_losses)), 4))

            self.peft_model.eval()
            for p in trainable_params:
                p.requires_grad = False

            student_dir = self._ruta_adaptador(sid)
            student_dir.mkdir(parents=True, exist_ok=True)
            # Guardar adaptador del estudiante en adapters_cache/{student_id}
            if sid == "default":
                self.peft_model.save_pretrained(str(student_dir))
            else:
                try:
                    self.peft_model.save_pretrained(
                        str(self.adapters_dir), selected_adapters=[sid]
                    )
                except TypeError:
                    self.peft_model.save_pretrained(str(student_dir))

            self.loaded_adapters.add(sid)
            if self.device == "cuda":
                torch.cuda.synchronize()

        elapsed = round(time.perf_counter() - t0, 3)
        loss_final = historial_loss[-1] if historial_loss else 0.0
        logger.info(
            "LoRA entrenado para %s | ejemplos=%d | epochs=%d | loss_final=%.4f | tiempo=%.2fs",
            sid, len(muestras), epochs, loss_final, elapsed,
        )
        return {
            "student_id": sid,
            "tiempo_entrenamiento_s": elapsed,
            "loss_final": loss_final,
            "historial_loss": historial_loss,
            "ejemplos_entrenados": len(muestras),
            "adapter_dir": str(student_dir),
        }

    def _decidir_mock(
        self,
        estrategia: str,
        estado_actual: Sequence[float],
        ejemplos_icl: List[Dict[str, Any]],
    ) -> int:
        """Política heurística/KNN determinista para pruebas en CPU con MOCK_LLM=1."""
        if estrategia == "ICL" and ejemplos_icl:
            # El ejemplo más cercano está al final de `ejemplos_icl`
            return int(ejemplos_icl[-1].get("accion", 0)) % 4

        izq, diag_izq, frente, diag_der, der = [float(x) for x in estado_actual[:5]]
        if frente < 0.22:
            return 3 if (diag_izq < 0.25 and diag_der < 0.25) else (1 if diag_izq > diag_der else 2)
        if diag_izq + 0.08 * izq > diag_der + 0.08 * der + 0.12:
            return 1
        if diag_der + 0.08 * der > diag_izq + 0.08 * izq + 0.12:
            return 2
        return 0

    def decidir_accion(
        self,
        student_id: str,
        estrategia: str,
        estado_actual: Sequence[float],
        historial: Sequence[int],
        jugadas_referencia: Optional[List[Dict[str, Any]]] = None,
        k_shots: int = 8,
        seleccion: str = "knn",
        pesos_sensores: Optional[Sequence[float]] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        precision_sensores: int = 50,
        ventana_historia: int = 3,
    ) -> Dict[str, Any]:
        """Predice la siguiente acción discreta {0,1,2,3} usando ICL o LoRA."""
        self.inicializar()
        sid = sanitizar_student_id(student_id)
        modo = estrategia.upper().strip()
        if modo not in {"ICL", "LORA"}:
            raise ValueError("estrategia debe ser 'ICL' o 'LORA'.")

        if k_shots > MAX_K_SHOTS:
            raise ValueError(f"k_shots={k_shots} excede el límite permitido ({MAX_K_SHOTS}).")

        ejemplos_icl: List[Dict[str, Any]] = []
        if modo == "ICL" and jugadas_referencia:
            ejemplos_icl = seleccionar_ejemplos_icl(
                estado_actual=estado_actual,
                jugadas_referencia=jugadas_referencia,
                k_shots=k_shots,
                seleccion=seleccion,
                pesos_sensores=pesos_sensores,
            )

        lineas = [SYSTEM_HEADER]
        if modo == "ICL":
            for ej in ejemplos_icl:
                lineas.append(
                    formatear_ejemplo(
                        ej,
                        precision=precision_sensores,
                        ventana_historia=ventana_historia,
                    )
                    + "\n"
                )
        query_str = formatear_estado(
            estado_actual,
            historial,
            precision=precision_sensores,
            ventana_historia=ventana_historia,
        )
        lineas.append(f"{query_str} ->")
        prompt = "".join(lineas)

        if self.mock_mode:
            t0 = time.perf_counter()
            if modo == "LORA":
                student_dir = self._ruta_adaptador(sid)
                if not student_dir.exists():
                    raise FileNotFoundError(
                        f"El estudiante '{sid}' aún no tiene un adaptador LoRA entrenado."
                    )
            accion = self._decidir_mock(modo, estado_actual, ejemplos_icl)
            lat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return {
                "accion": accion,
                "latencia_gpu_ms": lat_ms,
                "prompt_tokens": len(prompt.split()),
            }

        import torch

        t0 = time.perf_counter()
        with gpu_lock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_tokens = int(inputs["input_ids"].shape[1])

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": 2,
                "do_sample": float(temperature) > 0.0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if float(temperature) > 0.0:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)

            with torch.inference_mode():
                if modo == "ICL":
                    if self.peft_model is not None:
                        with self.peft_model.disable_adapter():
                            out_ids = self.peft_model.generate(**inputs, **gen_kwargs)
                    else:
                        out_ids = self.base_model.generate(**inputs, **gen_kwargs)
                else:
                    self._asegurar_adaptador_cargado(sid)
                    out_ids = self.peft_model.generate(**inputs, **gen_kwargs)

            new_tokens = out_ids[0][prompt_tokens:]
            texto_generado = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if self.device == "cuda":
                torch.cuda.synchronize()

        accion = parsear_accion(texto_generado, estado_actual)
        latencia_gpu_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "accion": accion,
            "latencia_gpu_ms": latencia_gpu_ms,
            "prompt_tokens": prompt_tokens,
        }


motor_global = MotorQwen()
