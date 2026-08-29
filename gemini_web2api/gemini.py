"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import urllib.request
import urllib.parse
import ssl
import os
import sys
import hashlib
import threading

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .config import CONFIG

_ssl_ctx = None
_httpx_client = None
_auth_lock = threading.RLock()
_log_lock = threading.Lock()
_runtime_metadata_lock = threading.Lock()
_runtime_metadata = {"auth_key": None, "tokens": {}}
_conversation_lock = threading.RLock()
_EMPTY_CONVERSATION_METADATA = ["", "", "", None, None, None, None, None, None, ""]
_DEFAULT_CONVERSATION_ID = "__default__"
_conversation_states = {}
_active_conversation_id = None
_conversation_store_path = None
_conversation_store_loaded = False
_auth_cache = {
    "path": None,
    "signature": None,
    "checked_signature": None,
    "data": {},
    "error": None,
}
_AUTH_FIELDS = {"auth_user", "xsrf_token", "gemini_bl"}


def log(msg: str):
    if not CONFIG["log_requests"]:
        return
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    if getattr(sys, "frozen", False):
        path = os.path.join(os.path.dirname(sys.executable), "gemini-web2api.log")
        try:
            with _log_lock, open(path, "a", encoding="utf-8") as file:
                file.write(line)
        except OSError:
            pass
    elif sys.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _get_httpx_client():
    global _httpx_client
    if _httpx_client is None and HAS_HTTPX:
        proxy = CONFIG.get("proxy")
        transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
        _httpx_client = httpx.Client(transport=transport, timeout=CONFIG["request_timeout_sec"], verify=True)
    return _httpx_client


def _base_auth() -> dict:
    return {
        "cookie": "",
        "sapisid": None,
        "auth_user": CONFIG.get("auth_user"),
        "xsrf_token": CONFIG.get("xsrf_token"),
        "gemini_bl": CONFIG["gemini_bl"],
    }


def _parse_auth_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError("auth file is empty")
    if not content.startswith("{"):
        pairs = dict(p.split("=", 1) for p in content.split("; ") if "=" in p)
        return {"cookie": content, "sapisid": pairs.get("SAPISID") or None}

    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("auth JSON must be an object")
    cookie = payload.get("cookie")
    if not isinstance(cookie, str):
        raise ValueError("auth JSON requires a cookie string")
    if not cookie.strip():
        return {}
    sapisid = payload.get("sapisid")
    if sapisid is not None and not isinstance(sapisid, str):
        raise ValueError("sapisid must be a string or null")
    if not sapisid:
        pairs = dict(p.split("=", 1) for p in cookie.split("; ") if "=" in p)
        sapisid = pairs.get("SAPISID") or None

    parsed = {"cookie": cookie, "sapisid": sapisid}
    for key in _AUTH_FIELDS:
        if key not in payload:
            continue
        value = payload[key]
        if key == "auth_user":
            if value is not None and not isinstance(value, (str, int)):
                raise ValueError("auth_user must be a string, integer, or null")
        elif value is not None and not isinstance(value, str):
            raise ValueError(f"{key} must be a string or null")
        parsed[key] = value
    return parsed


def refresh_auth(force: bool = False) -> dict:
    """Return an immutable-per-call auth snapshot, reloading changed files safely."""
    path = CONFIG.get("cookie_file")
    base = _base_auth()
    with _auth_lock:
        if not path:
            _auth_cache.update({
                "path": None, "signature": None, "checked_signature": None,
                "data": {}, "error": None,
            })
            return base
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            _auth_cache.update({
                "path": path, "signature": None, "checked_signature": None,
                "data": {}, "error": "file not found",
            })
            return base

        stat = os.stat(path)
        signature = (stat.st_mtime_ns, stat.st_size)
        same_path = _auth_cache["path"] == path
        if not same_path:
            _auth_cache.update({
                "path": path, "signature": None, "checked_signature": None,
                "data": {}, "error": None,
            })
        if force or not same_path or _auth_cache["checked_signature"] != signature:
            try:
                parsed = _parse_auth_file(path)
            except Exception as e:
                _auth_cache.update({
                    "path": path,
                    "checked_signature": signature,
                    "error": str(e),
                })
                log(f"Auth reload error: {e}")
            else:
                _auth_cache.update({
                    "path": path,
                    "signature": signature,
                    "checked_signature": signature,
                    "data": parsed,
                    "error": None,
                })
                log(f"Auth reloaded: {path}")
        if _auth_cache["path"] == path:
            base.update(_auth_cache["data"])
        return dict(base)


def auth_status() -> dict:
    """Return non-secret auth loader status for startup diagnostics."""
    snapshot = refresh_auth()
    path = CONFIG.get("cookie_file")
    with _auth_lock:
        return {
            "path": os.path.abspath(path) if path else None,
            "exists": bool(path and os.path.isfile(path)),
            "loaded": bool(snapshot["cookie"]),
            "error": _auth_cache["error"],
        }


def load_cookie() -> tuple:
    auth = refresh_auth()
    return auth["cookie"], auth["sapisid"]


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def _auth_key(auth: dict) -> tuple:
    return (
        str(auth.get("auth_user") or ""),
        hashlib.sha256(auth.get("cookie", "").encode()).hexdigest(),
    )


def set_runtime_page_metadata(auth: dict, tokens: dict) -> None:
    """Cache fresh non-cookie page metadata for the matching auth snapshot."""
    allowed = {key: tokens[key] for key in ("at", "bl", "session_id") if tokens.get(key)}
    with _runtime_metadata_lock:
        _runtime_metadata.update({"auth_key": _auth_key(auth), "tokens": allowed})


def _runtime_page_metadata(auth: dict) -> dict:
    with _runtime_metadata_lock:
        if _runtime_metadata["auth_key"] != _auth_key(auth):
            return {}
        return dict(_runtime_metadata["tokens"])


def _account_prefix(auth: dict = None) -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth = auth or refresh_auth()
    auth_user = auth.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


def _build_headers(auth: dict = None, request_uuid: str = None) -> dict:
    auth = auth or refresh_auth()
    account_prefix = _account_prefix(auth)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if request_uuid:
        headers["X-Goog-Ext-525005358-Jspb"] = json.dumps([request_uuid, 1])
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(auth["auth_user"])
    cookie_str, sapisid = auth["cookie"], auth["sapisid"]
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    return headers


def _conversation_key(conversation_id: str = None) -> str:
    return conversation_id or _DEFAULT_CONVERSATION_ID


def _conversation_file() -> str:
    configured = CONFIG.get("conversation_store_file")
    if configured:
        return os.path.abspath(configured)
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    return os.path.join(base, "gemini-conversations.json")


def _load_conversation_store() -> None:
    global _active_conversation_id, _conversation_store_loaded, _conversation_store_path
    path = _conversation_file()
    with _conversation_lock:
        if _conversation_store_loaded and _conversation_store_path == path:
            return
        _conversation_states.clear()
        _active_conversation_id = None
        _conversation_store_path = path
        _conversation_store_loaded = True
        try:
            with open(path, encoding="utf-8") as file:
                payload = json.load(file)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            log(f"Conversation store load error: {error}")
            return
        conversations = payload.get("conversations", {}) if isinstance(payload, dict) else {}
        active = payload.get("active_conversation_id") if isinstance(payload, dict) else None
        _active_conversation_id = active if isinstance(active, str) else None
        if not isinstance(conversations, dict):
            return
        for conversation_id, state in conversations.items():
            if not isinstance(conversation_id, str) or not isinstance(state, dict):
                continue
            auth_key = state.get("auth_key")
            metadata = state.get("metadata")
            if (isinstance(auth_key, list) and len(auth_key) == 2
                    and isinstance(metadata, list)):
                _conversation_states[conversation_id] = {
                    "auth_key": tuple(str(value) for value in auth_key),
                    "metadata": list(metadata),
                    "updated_at": int(state.get("updated_at") or 0),
                }


def _save_conversation_store() -> None:
    path = _conversation_store_path or _conversation_file()
    payload = {
        "version": 1,
        "active_conversation_id": _active_conversation_id,
        "conversations": {
            conversation_id: {
                "auth_key": list(state["auth_key"]),
                "metadata": state["metadata"],
                "updated_at": state.get("updated_at", 0),
            }
            for conversation_id, state in _conversation_states.items()
            if conversation_id != _DEFAULT_CONVERSATION_ID
        },
    }
    directory = os.path.dirname(path)
    temporary = path + ".tmp"
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    except OSError as error:
        log(f"Conversation store save error: {error}")
        try:
            os.remove(temporary)
        except OSError:
            pass


def create_conversation(conversation_id: str, metadata: list = None, auth: dict = None, select: bool = False) -> dict:
    _load_conversation_store()
    auth = auth or refresh_auth()
    values = list(metadata or _EMPTY_CONVERSATION_METADATA)
    if len(values) < len(_EMPTY_CONVERSATION_METADATA):
        values.extend(_EMPTY_CONVERSATION_METADATA[len(values):])
    global _active_conversation_id
    with _conversation_lock:
        _conversation_states[conversation_id] = {
            "auth_key": _auth_key(auth),
            "metadata": values,
            "updated_at": int(time.time()),
        }
        if select:
            _active_conversation_id = conversation_id
        _save_conversation_store()
    return conversation_info(conversation_id, auth)


def conversation_info(conversation_id: str, auth: dict = None) -> dict:
    _load_conversation_store()
    auth = auth or refresh_auth()
    with _conversation_lock:
        state = _conversation_states.get(_conversation_key(conversation_id))
        if not state or state["auth_key"] != _auth_key(auth):
            return None
        metadata = list(state["metadata"])
        return {
            "id": conversation_id,
            "active": conversation_id == _active_conversation_id,
            "metadata": metadata,
            "cid": metadata[0] if metadata else "",
            "rid": metadata[1] if len(metadata) > 1 else "",
            "rcid": metadata[2] if len(metadata) > 2 else "",
            "updated_at": state.get("updated_at", 0),
        }


def active_conversation_id() -> str:
    _load_conversation_store()
    with _conversation_lock:
        return _active_conversation_id


def select_conversation(conversation_id: str, auth: dict = None) -> dict:
    global _active_conversation_id
    info = conversation_info(conversation_id, auth)
    if not info:
        return None
    with _conversation_lock:
        _active_conversation_id = conversation_id
        _save_conversation_store()
    return conversation_info(conversation_id, auth)


def list_conversations(auth: dict = None) -> list:
    _load_conversation_store()
    auth = auth or refresh_auth()
    key = _auth_key(auth)
    with _conversation_lock:
        ids = [
            conversation_id for conversation_id, state in _conversation_states.items()
            if conversation_id != _DEFAULT_CONVERSATION_ID and state["auth_key"] == key
        ]
    return [conversation_info(conversation_id, auth) for conversation_id in sorted(ids)]


def reset_conversation(conversation_id: str = None) -> None:
    global _active_conversation_id
    _load_conversation_store()
    key = _conversation_key(conversation_id)
    with _conversation_lock:
        _conversation_states.pop(key, None)
        if conversation_id:
            if _active_conversation_id == conversation_id:
                _active_conversation_id = None
            _save_conversation_store()
    log(f"Gemini conversation reset: {conversation_id or 'default'}")


def _conversation_metadata(auth: dict, conversation_id: str = None) -> list:
    _load_conversation_store()
    key = _conversation_key(conversation_id)
    auth_key = _auth_key(auth)
    with _conversation_lock:
        state = _conversation_states.get(key)
        if not state or state["auth_key"] != auth_key:
            state = {
                "auth_key": auth_key,
                "metadata": list(_EMPTY_CONVERSATION_METADATA),
                "updated_at": int(time.time()),
            }
            _conversation_states[key] = state
        return list(state["metadata"])


def _update_conversation_metadata(raw: str, auth: dict, conversation_id: str = None) -> None:
    latest = None
    for line in raw.splitlines():
        if '"wrb.fr"' not in line:
            continue
        try:
            frame = json.loads(line)
            inner = json.loads(frame[0][2])
            metadata = inner[1]
            if isinstance(metadata, list) and metadata and metadata[0]:
                latest = metadata
        except (json.JSONDecodeError, IndexError, TypeError):
            continue
    if latest:
        _load_conversation_store()
        with _conversation_lock:
            _conversation_states[_conversation_key(conversation_id)] = {
                "auth_key": _auth_key(auth),
                "metadata": list(latest),
                "updated_at": int(time.time()),
            }
            if conversation_id:
                _save_conversation_store()


def _apply_chat_persistence_flags(inner: list) -> None:
    """Apply Gemini Web persistence flags to an outgoing request payload."""
    if CONFIG.get("temporary_chats", False):
        # Match Gemini Web temporary-chat requests.
        inner[41] = [1]
        inner[45] = 1
    else:
        inner[41] = [2]


def _build_payload(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, auth: dict = None, request_uuid: str = None, conversation_id: str = None) -> str:
    inner = [None] * 81
    if file_refs:
        refs = []
        for item in file_refs:
            if isinstance(item, tuple) and len(item) == 2:
                ref, filename = item
            else:
                ref, filename = item, "image.png"
            refs.append([[ref], filename])
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    auth = auth or refresh_auth()
    inner[2] = _conversation_metadata(auth, conversation_id)
    inner[4] = uuid.uuid4().hex
    inner[6] = [1]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[0]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    _apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[55] = [[1]]
    inner[59] = request_uuid or str(uuid.uuid4()).upper()
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    inner[80] = 2 if think_mode == 0 else 1
    if extra_fields:
        for k, v in extra_fields.items():
            inner[k] = v
    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    metadata = _runtime_page_metadata(auth)
    access_token = metadata.get("at") or auth.get("xsrf_token")
    if access_token:
        params["at"] = access_token
    return urllib.parse.urlencode(params)


def _get_url(auth: dict = None) -> str:
    auth = auth or refresh_auth()
    metadata = _runtime_page_metadata(auth)
    reqid = int(time.time()) % 1000000
    account_prefix = _account_prefix(auth)
    query = {
        "bl": metadata.get("bl") or auth["gemini_bl"],
        "hl": "en",
        "_reqid": reqid,
        "rt": "c",
    }
    if metadata.get("session_id"):
        query["f.sid"] = metadata["session_id"]
    return (
        f"https://gemini.google.com{account_prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate?"
        + urllib.parse.urlencode(query)
    )


def clean_text(text: str, strip: bool = True) -> str:
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'http://googleusercontent\.com/card_content/\d+\n?', '', text)
    return text.strip() if strip else text


def _extract_texts_from_line(line: str) -> list:
    """Parse a single wrb.fr line and return list of text strings found."""
    if '"wrb.fr"' not in line:
        return []
    try:
        arr = json.loads(line)
        inner_str = arr[0][2]
        if not inner_str:
            return []
        inner = json.loads(inner_str)
        if not (isinstance(inner, list) and len(inner) > 4 and inner[4]):
            return []
        texts = []
        for part in inner[4]:
            if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                for t in part[1]:
                    if isinstance(t, str) and t:
                        texts.append(t)
        return texts
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


class GeminiUpstreamError(RuntimeError):
    def __init__(self, code: int):
        self.code = code
        super().__init__(f"Gemini upstream rejected request: error {code}")


def _extract_upstream_error_code(raw: str):
    bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', raw)
    if bard_err:
        return int(bard_err.group(1))
    for line in raw.splitlines():
        if '"wrb.fr"' not in line:
            continue
        try:
            frame = json.loads(line)
            code = frame[0][5][2][0][1][0]
            if isinstance(code, int):
                return code
        except (json.JSONDecodeError, IndexError, TypeError):
            continue
    return None


def _log_upstream_rejection(error, raw: str, auth: dict, conversation_id: str,
                            prompt: str, file_refs: list) -> None:
    metadata = _conversation_metadata(auth, conversation_id)
    log(
        "Gemini rejection diagnostics: "
        f"code={error.code} conversation={conversation_id or _DEFAULT_CONVERSATION_ID} "
        f"upstream_thread={'yes' if all(metadata[:3]) else 'no'} "
        f"prompt_chars={len(prompt)} attachments={len(file_refs or [])} "
        f"auth={'authenticated' if auth.get('cookie') else 'anonymous'} "
        f"response={_response_shape(raw)}"
    )


def extract_response_text(raw: str) -> str:
    """Parse full response to get final text."""
    if error_code := _extract_upstream_error_code(raw):
        raise GeminiUpstreamError(error_code)
    last_text = ""
    for line in raw.split("\n"):
        for t in _extract_texts_from_line(line):
            if len(t) > len(last_text):
                last_text = t
    return clean_text(last_text)


def _response_shape(raw: str, limit: int = 120) -> str:
    """Return bounded response structure without exposing string contents."""
    entries = []

    def walk(value, path, depth=0):
        if len(entries) >= limit or depth > 8:
            return
        if isinstance(value, list):
            entries.append(f"{path}=list:{len(value)}")
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", depth + 1)
        elif isinstance(value, dict):
            entries.append(f"{path}=dict:{len(value)}")
            for index, item in enumerate(value.values()):
                walk(item, f"{path}.value[{index}]", depth + 1)
        elif isinstance(value, str):
            entries.append(f"{path}=str:{len(value)}")
            if value[:1] in "[{":
                try:
                    walk(json.loads(value), f"{path}<json>", depth + 1)
                except json.JSONDecodeError:
                    pass
        elif value is None:
            entries.append(f"{path}=null")
        elif isinstance(value, (bool, int, float)):
            entries.append(f"{path}={type(value).__name__}:{value}")
        else:
            entries.append(f"{path}={type(value).__name__}")

    frames = 0
    for line_number, line in enumerate(raw.splitlines()):
        if '"wrb.fr"' not in line:
            continue
        frames += 1
        try:
            walk(json.loads(line), f"line[{line_number}]")
        except json.JSONDecodeError:
            entries.append(f"line[{line_number}]=invalid-json:{len(line)}")
    return (
        f"bytes={len(raw)} lines={len(raw.splitlines())} wrb_frames={frames} "
        f"bard_error={'yes' if 'BardErrorInfo' in raw else 'no'} shape="
        + ";".join(entries)
    )


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, conversation_id: str = None) -> str:
    """Non-streaming generation with retry."""
    auth = refresh_auth()
    request_uuid = str(uuid.uuid4()).upper()
    body = _build_payload(
        prompt, model_id, think_mode, file_refs, extra_fields, auth, request_uuid,
        conversation_id,
    ).encode()
    url = _get_url(auth)
    headers = _build_headers(auth, request_uuid)
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")

    last_err = None
    raw = ""
    for attempt in range(CONFIG["retry_attempts"]):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx)
                )
                resp = opener.open(req, timeout=CONFIG["request_timeout_sec"])
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])
            raw = resp.read().decode("utf-8", errors="replace")
            text = extract_response_text(raw)
            _update_conversation_metadata(raw, auth, conversation_id)
            if not text:
                log(f"Empty Gemini response structure: {_response_shape(raw)}")
                raise RuntimeError("Gemini upstream returned an empty response")
            return text
        except Exception as e:
            last_err = e
            if isinstance(e, GeminiUpstreamError):
                _log_upstream_rejection(
                    e, raw, auth, conversation_id, prompt, file_refs,
                )
                break
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, conversation_id: str = None):
    """Streaming generation via httpx with retry on connection failure."""
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields, conversation_id)
        if text:
            yield text
        return

    auth = refresh_auth()
    request_uuid = str(uuid.uuid4()).upper()
    body = _build_payload(
        prompt, model_id, think_mode, file_refs, extra_fields, auth, request_uuid,
        conversation_id,
    )
    url = _get_url(auth)
    headers = _build_headers(auth, request_uuid)
    client = _get_httpx_client()

    last_err = None
    emitted_raw_text = ""
    rejection_raw = ""
    for attempt in range(CONFIG["retry_attempts"]):
        try:
            attempt_emitted = False
            with client.stream("POST", url, content=body, headers=headers) as resp:
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    rejection_raw = buf
                    if "BardErrorInfo" in buf:
                        bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', buf)
                        if bard_err:
                            raise GeminiUpstreamError(int(bard_err.group(1)))
                    lines = []
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        lines.append(line)
                    for line in lines:
                        _update_conversation_metadata(line, auth, conversation_id)
                        for t in _extract_texts_from_line(line):
                            if t == emitted_raw_text or emitted_raw_text.startswith(t):
                                continue
                            if not t.startswith(emitted_raw_text):
                                raise RuntimeError("Gemini stream content changed during retry")
                            delta = clean_text(t[len(emitted_raw_text):], strip=False)
                            emitted_raw_text = t
                            if delta:
                                attempt_emitted = True
                                yield delta
                _update_conversation_metadata(buf, auth, conversation_id)
                for t in _extract_texts_from_line(buf):
                    if t == emitted_raw_text or emitted_raw_text.startswith(t):
                        continue
                    if not t.startswith(emitted_raw_text):
                        raise RuntimeError("Gemini stream content changed during retry")
                    delta = clean_text(t[len(emitted_raw_text):], strip=False)
                    emitted_raw_text = t
                    if delta:
                        attempt_emitted = True
                        yield delta
            if not attempt_emitted and not emitted_raw_text:
                raise RuntimeError("Gemini upstream returned an empty response")
            return
        except Exception as e:
            last_err = e
            if isinstance(e, GeminiUpstreamError):
                _log_upstream_rejection(
                    e, rejection_raw, auth, conversation_id, prompt, file_refs,
                )
                break
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Stream retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err
