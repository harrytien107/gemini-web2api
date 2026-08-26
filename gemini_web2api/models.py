"""Model definitions and mapping from Gemini frontend JS source."""

import threading

from .config import CONFIG


_round_robin_lock = threading.Lock()
# ponytail: Rotation is per process; use shared storage only when coordinating multiple processes.
_round_robin_positions = {}

# MODE_CATEGORY enum from 028-6eb337387583.js:
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE

MODELS = {
    "gemini-3.7-flash": {
        "mode": 1, "think": 4,
        "desc": "Latest all-around model (Gemini 3.7 Flash)",
    },
    "gemini-3.7-flash-thinking": {
        "mode": 2, "think": 0,
        "desc": "Latest deep-thinking model (Gemini 3.7 Flash)",
    },
    "gemini-3.6-flash": {
        "mode": 1, "think": 4,
        "desc": "All-around model (Gemini 3.6 Flash)",
    },
    "gemini-3.6-flash-thinking": {
        "mode": 2, "think": 0,
        "desc": "Deep-thinking model (Gemini 3.6 Flash)",
    },
    "gemini-3.5-flash": {
        "mode": 1, "think": 4,
        "desc": "Alias for gemini-3.6-flash (backend upgraded)",
    },
    "gemini-3.5-flash-thinking": {
        "mode": 2, "think": 0,
        "desc": "Deep thinking mode, longest output (~20k chars)",
    },
    "gemini-3.1-pro": {
        "mode": 3, "think": 4,
        "desc": "Pro model (requires cookie for real routing)",
    },
    "gemini-3.1-pro-enhanced": {
        "mode": 3, "think": 4, "extra": {31: 2, 80: 3},
        "desc": "Pro with enhanced output (experimental)",
    },
    "gemini-auto": {
        "mode": 4, "think": 4,
        "desc": "Auto model selection",
    },
    "gemini-3.5-flash-thinking-lite": {
        "mode": 5, "think": 0,
        "desc": "Dynamic thinking with adaptive depth",
    },
    "gemini-flash-lite": {
        "mode": 6, "think": 4,
        "desc": "Lightweight fast model",
    },
}


def resolve_model(model_name: str):
    """Resolve model name to (name, mode_id, think_mode, error, extra_fields)."""
    think_override = None
    if "@think=" in model_name:
        model_name, think_str = model_name.rsplit("@think=", 1)
        try:
            think_override = int(think_str)
        except ValueError:
            return None, None, None, f"Invalid think level: {think_str}", None
    cfg = MODELS.get(model_name)
    if not cfg:
        return None, None, None, f"Unknown model: {model_name}", None
    mode_id = cfg["mode"]
    think_mode = think_override if think_override is not None else cfg["think"]
    extra = cfg.get("extra")
    return model_name, mode_id, think_mode, None, extra


def configured_model_combos() -> dict:
    """Return configured named model groups."""
    combos = CONFIG.get("model_combos") or {}
    return dict(combos) if isinstance(combos, dict) else {}


def resolve_model_chain(model_name: str):
    """Resolve a model or named combo into ordered (name, mode, think, extra) attempts."""
    combos = configured_model_combos()
    if model_name not in combos:
        name, mode_id, think_mode, error, extra = resolve_model(model_name)
        return name, [(name, mode_id, think_mode, extra)] if not error else [], error

    configured = combos[model_name]
    if isinstance(configured, list):
        strategy, configured_models = "fallback", configured
    elif isinstance(configured, dict):
        strategy = configured.get("strategy", "fallback")
        configured_models = configured.get("models")
    else:
        return model_name, [], f"{model_name} is not configured"

    if strategy not in ("fallback", "round_robin"):
        return model_name, [], f"Invalid combo strategy: {strategy!r}"
    if not isinstance(configured_models, list) or not configured_models:
        return model_name, [], f"{model_name} is not configured"

    attempts = []
    for item in configured_models:
        if not isinstance(item, str) or not item or item in combos:
            return model_name, [], f"Invalid combo model: {item!r}"
        base_name = item.rsplit("@think=", 1)[0]
        if base_name not in MODELS:
            return model_name, [], f"Unknown combo model: {base_name}"
        name, mode_id, think_mode, error, extra = resolve_model(item)
        if error:
            return model_name, [], error
        attempts.append((name, mode_id, think_mode, extra))

    if strategy == "round_robin":
        with _round_robin_lock:
            start = _round_robin_positions.get(model_name, 0) % len(attempts)
            _round_robin_positions[model_name] = start + 1
        attempts = attempts[start:] + attempts[:start]
    return model_name, attempts, None
