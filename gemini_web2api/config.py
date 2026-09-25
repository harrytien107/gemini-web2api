"""Configuration management."""
import json
import os
import tempfile
import threading

DEFAULT_CONFIG = {
    "port": 8081,
    "host": "0.0.0.0",
    "retry_attempts": 3,
    "retry_delay_sec": 2,
    "request_timeout_sec": 180,
    "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
    "auth_user": None,
    "xsrf_token": None,
    "model_definitions": {},
    "model_combos": {},
    "log_requests": True,
    "cookie_file": None,
    "proxy": None,
    "api_keys": [],
    "temporary_chats": False,
    "conversation_store_file": None,
}

CONFIG = dict(DEFAULT_CONFIG)
CONFIG_PATH = None
_CONFIG_LOCK = threading.RLock()


def load_config(path: str = None):
    """Load config from JSON file and remember its path for runtime updates."""
    global CONFIG_PATH
    with _CONFIG_LOCK:
        if not path:
            return CONFIG
        resolved = os.path.abspath(path)
        CONFIG_PATH = resolved
        if not os.path.exists(resolved):
            return CONFIG
        with open(resolved, encoding="utf-8") as file:
            loaded = json.load(file)
        if not isinstance(loaded, dict):
            raise ValueError("config root must be a JSON object")
        CONFIG.update(loaded)
        return CONFIG


def save_config_updates(updates: dict, path: str = None):
    """Atomically persist selected config keys while preserving unrelated settings."""
    global CONFIG_PATH
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dictionary")
    with _CONFIG_LOCK:
        target = path or CONFIG_PATH or find_config() or "./config.json"
        target = os.path.abspath(target)
        current = {}
        if os.path.exists(target):
            with open(target, encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, dict):
                raise ValueError("config root must be a JSON object")
            current.update(loaded)
        current.update(updates)
        directory = os.path.dirname(target) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(current, file, indent=2, ensure_ascii=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, target)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
        CONFIG.update(updates)
        CONFIG_PATH = target
        return dict(current)


def find_config():
    """Search for config file in standard locations."""
    for p in ["./config.json", os.path.expanduser("~/.config/gemini-web2api/config.json")]:
        if os.path.exists(p):
            return p
    return None
