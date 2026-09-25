"""Model definitions and mapping from Gemini frontend JS source."""

import re
import threading

from .config import CONFIG, save_config_updates


_round_robin_lock = threading.Lock()
_model_config_lock = threading.RLock()
_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# ponytail: Rotation is per process; use shared storage only when coordinating multiple processes.
_round_robin_positions = {}

# MODE_CATEGORY enum from 028-6eb337387583.js:
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE

MODELS = {
    "gemini-3.8-flash": {
        "mode": 1, "think": 4,
        "desc": "Latest all-around model (Gemini 3.8 Flash)",
    },
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


def configured_model_definitions() -> dict:
    """Return persisted raw model additions and built-in overrides."""
    definitions = CONFIG.get("model_definitions") or {}
    return dict(definitions) if isinstance(definitions, dict) else {}


def effective_model_definitions() -> dict:
    """Merge immutable defaults with persisted additions and overrides."""
    catalog = {name: dict(config) for name, config in MODELS.items()}
    for name, config in configured_model_definitions().items():
        try:
            catalog[name] = validate_model_definition(name, config)
        except ValueError:
            continue
    return catalog


def validate_model_definition(name: str, config) -> dict:
    """Validate and normalize one raw Gemini frontend model definition."""
    if not isinstance(name, str) or not _MODEL_NAME_PATTERN.fullmatch(name.strip()):
        raise ValueError(
            "model name must be 1-128 characters using letters, numbers, '.', '_', ':', or '-'"
        )
    name = name.strip()
    if name in configured_model_combos():
        raise ValueError(f"model name conflicts with custom combo {name!r}")
    if not isinstance(config, dict):
        raise ValueError("model definition must be an object")
    mode = config.get("mode")
    think = config.get("think")
    desc = config.get("desc", config.get("description", ""))
    extra = config.get("extra")
    if isinstance(mode, bool) or not isinstance(mode, int) or not 1 <= mode <= 6:
        raise ValueError("mode must be an integer from 1 to 6")
    if isinstance(think, bool) or not isinstance(think, int) or not 0 <= think <= 4:
        raise ValueError("think must be an integer from 0 to 4")
    if not isinstance(desc, str) or len(desc) > 500:
        raise ValueError("description must be a string up to 500 characters")
    normalized = {"mode": mode, "think": think, "desc": desc.strip()}
    if extra is not None:
        if not isinstance(extra, dict):
            raise ValueError("extra must be an object mapping payload indexes to values")
        normalized_extra = {}
        for key, value in extra.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                raise ValueError(f"extra index must be an integer: {key!r}")
            if isinstance(key, float) or not 0 <= index <= 80:
                raise ValueError("extra indexes must be integers from 0 to 80")
            normalized_extra[index] = value
        normalized["extra"] = normalized_extra
    return normalized


def save_model_definition(name: str, config) -> dict:
    """Persist one raw model addition or built-in override."""
    normalized = validate_model_definition(name, config)
    name = name.strip()
    with _model_config_lock:
        definitions = configured_model_definitions()
        definitions[name] = normalized
        save_config_updates({"model_definitions": definitions})
    return {
        "name": name,
        **normalized,
        "default": name in MODELS,
        "customized": True,
    }


def reset_model_definition(name: str) -> bool:
    """Remove a raw addition or restore a built-in to its hardcoded default."""
    if not isinstance(name, str) or not _MODEL_NAME_PATTERN.fullmatch(name.strip()):
        raise ValueError(
            "model name must be 1-128 characters using letters, numbers, '.', '_', ':', or '-'"
        )
    name = name.strip()
    with _model_config_lock:
        definitions = configured_model_definitions()
        if name not in definitions:
            return False
        definitions.pop(name)
        save_config_updates({"model_definitions": definitions})
    return True


def resolve_model(model_name: str):
    """Resolve model name to (name, mode_id, think_mode, error, extra_fields)."""
    think_override = None
    if "@think=" in model_name:
        model_name, think_str = model_name.rsplit("@think=", 1)
        try:
            think_override = int(think_str)
        except ValueError:
            return None, None, None, f"Invalid think level: {think_str}", None
    cfg = effective_model_definitions().get(model_name)
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


def validate_model_combo(name: str, config) -> dict:
    """Validate and normalize a custom model combo."""
    if not isinstance(name, str) or not _MODEL_NAME_PATTERN.fullmatch(name.strip()):
        raise ValueError(
            "model combo name must be 1-128 characters using letters, numbers, '.', '_', ':', or '-'"
        )
    name = name.strip()
    if name in effective_model_definitions():
        raise ValueError(f"model combo name conflicts with model {name!r}")

    if isinstance(config, list):
        strategy, models = "fallback", config
    elif isinstance(config, dict):
        strategy = config.get("strategy", "fallback")
        models = config.get("models")
    else:
        raise ValueError("model combo must be an object with strategy and models")

    if strategy not in ("fallback", "round_robin"):
        raise ValueError("strategy must be 'fallback' or 'round_robin'")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")

    normalized_models = []
    combos = configured_model_combos()
    for item in models:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("each combo model must be a non-empty string")
        item = item.strip()
        base_name = item.rsplit("@think=", 1)[0]
        if base_name in combos and base_name != name:
            raise ValueError(f"nested custom model combos are not supported: {base_name}")
        if base_name not in effective_model_definitions():
            raise ValueError(f"unknown model: {base_name}")
        _, _, _, error, _ = resolve_model(item)
        if error:
            raise ValueError(error)
        normalized_models.append(item)

    return {"strategy": strategy, "models": normalized_models}


def save_model_combo(name: str, config) -> dict:
    """Validate and atomically persist one custom model combo."""
    normalized = validate_model_combo(name, config)
    name = name.strip()
    with _model_config_lock:
        combos = configured_model_combos()
        combos[name] = normalized
        save_config_updates({"model_combos": combos})
        with _round_robin_lock:
            _round_robin_positions.pop(name, None)
    return {"name": name, **normalized}


def delete_model_combo(name: str) -> bool:
    """Delete a persisted custom model combo. Built-in models are immutable."""
    if not isinstance(name, str) or not _MODEL_NAME_PATTERN.fullmatch(name.strip()):
        raise ValueError(
            "model combo name must be 1-128 characters using letters, numbers, '.', '_', ':', or '-'"
        )
    name = name.strip()
    with _model_config_lock:
        combos = configured_model_combos()
        if name not in combos:
            return False
        combos.pop(name, None)
        save_config_updates({"model_combos": combos})
        with _round_robin_lock:
            _round_robin_positions.pop(name, None)
    return True


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
        if base_name not in effective_model_definitions():
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