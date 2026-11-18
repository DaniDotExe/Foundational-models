"""Reglas del reto: rangos permitidos y presupuesto de interacción."""
from .pistas import ORDEN_PISTAS

LIMITES = {
    "pasos_totales": (20_000, 600_000),
    "pasos_por_rollout": (128, 1024),
    "epocas": (1, 10),
    "tam_minibatch": (32, 2048),
    "tasa_aprendizaje": (5e-5, 3e-3),
    "gamma": (0.90, 0.999),
    "lam": (0.80, 1.0),
    "epsilon_clip": (0.05, 0.40),
    "coef_valor": (0.10, 1.0),
    "coef_entropia": (0.0, 0.05),
    "evaluar_cada": (1, 50),
}
PRESUPUESTO_MAX_PASOS = 600_000
NUM_PISTAS = len(ORDEN_PISTAS)


def validar_hiperparametros(cfg):
    errores = []
    for nombre, (a, b) in LIMITES.items():
        if nombre not in cfg:
            errores.append(f"falta el hiperparámetro '{nombre}'.")
            continue
        valor = cfg[nombre]
        if not isinstance(valor, (int, float)) or isinstance(valor, bool):
            errores.append(f"{nombre}={valor!r} debe ser un número.")
        elif not a <= valor <= b:
            errores.append(f"{nombre}={valor} debe estar entre {a} y {b}.")
    for nombre in ("pasos_totales", "pasos_por_rollout", "epocas", "tam_minibatch", "evaluar_cada", "semilla"):
        if nombre in cfg and not isinstance(cfg[nombre], int):
            errores.append(f"{nombre} debe ser un entero.")
    if "inicio_aleatorio" not in cfg or not isinstance(cfg["inicio_aleatorio"], bool):
        errores.append("inicio_aleatorio debe ser True o False.")
    if "semilla" not in cfg:
        errores.append("falta la semilla.")
    if not errores:
        lote = cfg["pasos_por_rollout"] * NUM_PISTAS
        if cfg["tam_minibatch"] > lote:
            errores.append(f"tam_minibatch={cfg['tam_minibatch']} no puede superar el lote de {lote} pasos "
                           f"(pasos_por_rollout x {NUM_PISTAS} pistas).")
        if cfg["pasos_totales"] < lote:
            errores.append(f"pasos_totales debe permitir al menos una actualización (pasos_por_rollout x {NUM_PISTAS}).")
    if errores:
        raise ValueError("Configuración no permitida:\n- " + "\n- ".join(errores))
    return True


def resumen_reglas():
    print("Limites:")
    for nombre, (a, b) in LIMITES.items():
        print(f"  {nombre:18s}: {a} .. {b}")
    print(f"  presupuesto maximo  : {PRESUPUESTO_MAX_PASOS} pasos de interaccion con el entorno")
