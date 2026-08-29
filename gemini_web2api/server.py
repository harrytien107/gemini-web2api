"""HTTP server: OpenAI-compatible API endpoints."""
import hashlib
import json
import time
import uuid
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import unquote

from .config import CONFIG
from .models import MODELS, configured_model_combos, resolve_model_chain
from .gemini import (
    GeminiUpstreamError,
    active_conversation_id,
    conversation_info,
    create_conversation,
    generate,
    generate_stream,
    list_conversations,
    log,
    refresh_auth,
    reset_conversation,
    select_conversation,
)
from .tools import messages_to_prompt, parse_tool_calls, google_contents_to_prompt, parse_google_function_calls
from .multimodal import detect_image_mime, fetch_image_bytes, upload_image
from . import __version__


_chat_history_lock = threading.Lock()
_chat_history = []
_chat_histories = {}
_CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CONVERSATION_MANAGER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemini Conversations</title><style>
:root{color-scheme:dark;font:16px system-ui;background:#111827;color:#e5e7eb}body{max-width:760px;margin:40px auto;padding:0 16px}h1{margin-bottom:4px}.muted{color:#9ca3af}section{background:#1f2937;border:1px solid #374151;border-radius:12px;padding:18px;margin:18px 0}input,button{font:inherit;border-radius:7px;border:1px solid #4b5563;padding:9px;background:#111827;color:#e5e7eb}input{box-sizing:border-box;width:100%;margin:5px 0 12px}button{cursor:pointer;background:#2563eb;border-color:#2563eb;margin:3px}button.danger{background:#991b1b;border-color:#991b1b}button.secondary{background:#374151;border-color:#4b5563}.row{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 0;border-top:1px solid #374151}.row:first-child{border-top:0}.id{font-weight:650;overflow-wrap:anywhere}.active{color:#86efac;font-size:.85em}#status{min-height:24px;color:#fbbf24}code{background:#111827;padding:2px 5px;border-radius:4px}</style></head>
<body><h1>Gemini Conversations</h1><p class="muted">Create or select the session used by clients that do not send <code>X-Conversation-ID</code>.</p>
<section><label>Local API key <span class="muted">(leave empty when api_keys is [])</span></label><input id="key" type="password" autocomplete="off" placeholder="Optional"><button class="secondary" onclick="saveKey()">Save key in this browser</button></section>
<section><label>New conversation name</label><input id="newId" maxlength="128" placeholder="my-overlay-chat"><button onclick="createConversation()">Create and select</button></section>
<section><div id="status"></div><div id="list">Loading…</div></section>
<script>
const key=document.querySelector('#key'),statusEl=document.querySelector('#status'),list=document.querySelector('#list');key.value=localStorage.getItem('gemini-api-key')||'';
function headers(json=false){const h={};if(json)h['Content-Type']='application/json';if(key.value)h.Authorization='Bearer '+key.value;return h}
function saveKey(){localStorage.setItem('gemini-api-key',key.value);load()}
async function request(path,options={}){options.headers={...headers(Boolean(options.body)),...(options.headers||{})};const response=await fetch(path,options);let data;try{data=await response.json()}catch{throw Error('Invalid server response')}if(!response.ok)throw Error(data.error?.message||data.error||('HTTP '+response.status));return data}
function setStatus(message,error=false){statusEl.textContent=message;statusEl.style.color=error?'#fca5a5':'#86efac'}
async function load(){try{const data=await request('/v1/conversations');const items=data.data||[];list.innerHTML=items.length?'':'<p class="muted">No saved conversations.</p>';for(const item of items){const row=document.createElement('div');row.className='row';const label=document.createElement('div');label.innerHTML='<div class="id"></div>'+(item.active?'<div class="active">Active</div>':'');label.querySelector('.id').textContent=item.id;const actions=document.createElement('div');const select=document.createElement('button');select.textContent='Select';select.disabled=item.active;select.onclick=()=>act(item.id,'select');const reset=document.createElement('button');reset.textContent='Reset';reset.className='danger';reset.onclick=()=>{if(confirm('Reset and remove '+item.id+'?'))act(item.id,'reset')};actions.append(select,reset);row.append(label,actions);list.append(row)}setStatus('')}catch(error){list.innerHTML='';setStatus(error.message,true)}}
async function createConversation(){const id=document.querySelector('#newId').value.trim();if(!id)return setStatus('Enter a conversation name.',true);try{await request('/v1/conversations',{method:'POST',body:JSON.stringify({id})});document.querySelector('#newId').value='';setStatus('Created and selected '+id);await load()}catch(error){setStatus(error.message,true)}}
async function act(id,action){try{await request('/v1/conversations/'+encodeURIComponent(id)+'/'+action,{method:'POST',body:'{}'});setStatus((action==='select'?'Selected ':'Reset ')+id);await load()}catch(error){setStatus(error.message,true)}}
load();
</script></body></html>"""


def _validate_conversation_id(conversation_id) -> str:
    if not isinstance(conversation_id, str) or not _CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
        raise ValueError("conversation_id must be 1-128 characters using letters, numbers, '.', '_', ':', or '-'")
    return conversation_id


def _incremental_chat_messages(messages: list, conversation_id: str = None) -> list:
    """Return only unseen messages; explicit conversation IDs own their reset lifecycle."""
    if not isinstance(messages, list):
        return []
    with _chat_history_lock:
        previous = list(_chat_histories.get(conversation_id, [])) if conversation_id else list(_chat_history)
    continues = bool(previous) and len(messages) >= len(previous) and messages[:len(previous)] == previous
    if previous and not continues:
        if not conversation_id:
            reset_conversation()
            return messages
        # Preserve the selected upstream session when an overlay rebuilds history on model change.
        last_assistant = max(
            (index for index, message in enumerate(messages)
             if isinstance(message, dict) and message.get("role") == "assistant"),
            default=-1,
        )
        return [
            message for message in messages[last_assistant + 1:]
            if isinstance(message, dict) and message.get("role", "user") != "assistant"
        ]
    if not continues:
        return messages
    new_messages = messages[len(previous):]
    return [message for message in new_messages if message.get("role", "user") != "assistant"]


def _commit_chat_messages(messages: list, conversation_id: str = None) -> None:
    global _chat_history
    with _chat_history_lock:
        if conversation_id:
            _chat_histories[conversation_id] = list(messages)
        else:
            _chat_history = list(messages)


def _clear_chat_history(conversation_id: str) -> None:
    with _chat_history_lock:
        _chat_histories.pop(conversation_id, None)


def _usage(prompt: str, text: str) -> dict:
    p = len(prompt) // 4
    c = len(text or "") // 4
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _model_catalog() -> dict:
    catalog = dict(MODELS)
    for name, configured in configured_model_combos().items():
        if isinstance(configured, list):
            strategy, models = "fallback", configured
        elif isinstance(configured, dict):
            strategy = configured.get("strategy", "fallback")
            models = configured.get("models")
        else:
            continue
        if (isinstance(name, str) and name and isinstance(models, list) and models
                and strategy in ("fallback", "round_robin")):
            catalog[name] = {"desc": f"Configured model {strategy} chain"}
    return catalog


def _generate_attempts(prompt, attempts, file_refs, combo, conversation_id=None):
    errors = []
    terminal_errors = []
    for name, model_id, think_mode, extra_fields in attempts:
        log(f"Model {name}: calling")
        caught = None
        try:
            text = generate(prompt, model_id, think_mode, file_refs, extra_fields, conversation_id)
            if text:
                log(f"Model {name}: success")
                return text
            error = "empty response"
        except Exception as e:
            caught = e
            error = str(e)
        log(f"Model {name}: failed ({error})")
        errors.append(f"{name}: {error}")
        if isinstance(caught, GeminiUpstreamError):
            terminal_errors.append(caught)
        if not combo:
            raise caught if caught is not None else RuntimeError(error)
    if len(terminal_errors) == len(errors) and terminal_errors:
        raise terminal_errors[-1]
    raise RuntimeError("all combo models failed: " + "; ".join(errors))


def _generate_stream_attempts(prompt, attempts, file_refs, combo, conversation_id=None):
    errors = []
    terminal_errors = []
    for name, model_id, think_mode, extra_fields in attempts:
        emitted = False
        caught = None
        log(f"Model {name}: calling")
        try:
            for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields, conversation_id):
                if delta:
                    emitted = True
                    yield delta
            if emitted:
                log(f"Model {name}: success")
                return
            error = "empty response"
        except Exception as e:
            if emitted:
                log(f"Model {name}: failed after output started ({e})")
                raise
            caught = e
            error = str(e)
        log(f"Model {name}: failed ({error})")
        errors.append(f"{name}: {error}")
        if isinstance(caught, GeminiUpstreamError):
            terminal_errors.append(caught)
        if not combo:
            raise caught if caught is not None else RuntimeError(error)
    if len(terminal_errors) == len(errors) and terminal_errors:
        raise terminal_errors[-1]
    raise RuntimeError("all combo models failed: " + "; ".join(errors))


class ImageAuthenticationRequired(ValueError):
    pass


def _upstream_error_response(error: Exception, has_images: bool) -> tuple[str, int]:
    if has_images and isinstance(error, GeminiUpstreamError) and error.code == 1100:
        return (
            "Gemini rejected the image because the exported Google session is expired or unauthenticated. "
            "Open Gemini while signed in, refresh the page, export a new gemini-auth.json, replace the file "
            "beside gemini-web2api-cookie.exe, then retry.",
            401,
        )
    return f"upstream error: {error}", 502


def _upload_images(images: list) -> list:
    """Upload each unique image once and return Gemini file references."""
    if not images:
        return None
    if not refresh_auth().get("cookie"):
        log(f"Image batch rejected: count={len(images)} auth=anonymous")
        raise ImageAuthenticationRequired(
            "image input requires authenticated Gemini cookies; use gemini-web2api-cookie.exe"
        )

    unique_images = []
    seen_sources = set()
    seen_content = set()
    for item in images:
        if not (isinstance(item, tuple) and len(item) == 2):
            continue
        data, mime = item
        if isinstance(data, str):
            if data in seen_sources:
                continue
            seen_sources.add(data)
            data = fetch_image_bytes(data)
            mime = mime or "image/png"
        if not data:
            raise RuntimeError("image fetch failed")
        fingerprint = hashlib.sha256(data).digest()
        if fingerprint in seen_content:
            continue
        seen_content.add(fingerprint)
        unique_images.append((data, mime))

    log(f"Image batch: count={len(images)} unique={len(unique_images)} auth=authenticated")
    file_refs = []
    for data, mime in unique_images:
        mime = detect_image_mime(data, mime or "image/png")
        try:
            filename = "image.png"
            ref = upload_image(data, filename, mime or "image/png")
            file_refs.append((ref, filename))
        except Exception as e:
            raise RuntimeError(f"image upload failed: {e}") from e
    return file_refs if file_refs else None


class GeminiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        client_ip = self.client_address[0] if self.client_address else "-"
        log(f"{client_ip} {fmt % args}")

    def send_html(self, text, status=200):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200, conversation_id=None):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "X-Conversation-ID")
        if conversation_id:
            self.send_header("X-Conversation-ID", conversation_id)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self, conversation_id=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "X-Conversation-ID")
        if conversation_id:
            self.send_header("X-Conversation-ID", conversation_id)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _write_sse(self, data):
        if isinstance(data, str):
            data = data.encode()
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _finish_sse(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _parse_body(self, body: bytes) -> dict:
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    def _read_request_body(self) -> bytes:
        transfer_encoding = self.headers.get("Transfer-Encoding", "")
        if "chunked" in transfer_encoding.lower():
            chunks = []
            while True:
                size_line = self.rfile.readline()
                if not size_line:
                    break
                size_text = size_line.split(b";", 1)[0].strip()
                try:
                    size = int(size_text, 16)
                except ValueError:
                    raise ValueError("invalid chunked request body")
                if size == 0:
                    while True:
                        trailer = self.rfile.readline()
                        if trailer in (b"\r\n", b"\n", b""):
                            break
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.read(2)
            return b"".join(chunks)

        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _request_conversation_id(self, req=None):
        conversation_id = self.headers.get("X-Conversation-ID")
        if not conversation_id and isinstance(req, dict):
            conversation_id = req.get("conversation_id") or req.get("conversationId")
            if not conversation_id and isinstance(req.get("conversation"), str):
                conversation_id = req["conversation"]
        conversation_id = conversation_id or active_conversation_id()
        return _validate_conversation_id(conversation_id) if conversation_id else None

    def _authorized(self):
        keys = CONFIG.get("api_keys") or []
        if not keys:
            return True
        # Authorization: Bearer <key>
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[7:] in keys:
            return True
        # header keys (OpenAI x-api-key / Google x-goog-api-key)
        for h in ("x-api-key", "x-goog-api-key"):
            if self.headers.get(h, "") in keys:
                return True
        # query param ?key= (Gemini CLI native style)
        if "?" in self.path:
            for pair in self.path.split("?", 1)[1].split("&"):
                if pair.startswith("key=") and pair[4:] in keys:
                    return True
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        try:
            refresh_auth()
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            path = self.path.split("?", 1)[0]
            if path == "/conversations":
                self.send_html(_CONVERSATION_MANAGER_HTML)
            elif path == "/v1/models":
                self.send_json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": 1700000000,
                     "owned_by": "google", "description": c["desc"]}
                    for n, c in _model_catalog().items()
                ]})
            elif path == "/v1/conversations":
                self.send_json({"object": "list", "data": list_conversations()})
            elif path.startswith("/v1/conversations/"):
                conversation_id = _validate_conversation_id(unquote(path[len("/v1/conversations/"):]))
                info = conversation_info(conversation_id)
                self.send_json(info or {"error": {"message": "conversation not found"}}, 200 if info else 404)
            elif path.startswith("/v1beta/models"):
                self.send_json({"models": [
                    {"name": f"models/{n}", "displayName": n, "description": c["desc"],
                     "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]}
                    for n, c in _model_catalog().items()
                ]})
            elif path == "/":
                self.send_json({"status": "ok", "version": __version__, "models": list(MODELS.keys())})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as error:
            self.send_json({"error": {"message": str(error)}}, 400)

    def do_POST(self):
        try:
            refresh_auth()
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            body = self._read_request_body()
            path = self.path.split("?", 1)[0]
            if path == "/v1/chat/completions":
                self._handle_chat(body)
            elif path == "/v1/responses":
                self._handle_responses(body)
            elif path == "/v1/conversations":
                self._handle_conversation_create(body)
            elif path.startswith("/v1/conversations/") and path.endswith("/select"):
                self._handle_conversation_select(path)
            elif path.startswith("/v1/conversations/") and path.endswith("/reset"):
                self._handle_conversation_reset(path)
            elif path.startswith("/v1/conversations/") and path.endswith("/attach"):
                self._handle_conversation_attach(path, body)
            elif ":streamGenerateContent" in self.path:
                self._handle_google_generate(body, stream=True)
            elif ":generateContent" in self.path:
                self._handle_google_generate(body, stream=False)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as error:
            self.send_json({"error": {"message": str(error)}}, 400)
        except Exception as e:
            log(f"POST error: {e}")
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except:
                pass

    def _conversation_path_id(self, path: str, action: str) -> str:
        prefix = "/v1/conversations/"
        return _validate_conversation_id(unquote(path[len(prefix):-len(action)]).rstrip("/"))

    def _handle_conversation_create(self, body: bytes):
        req = self._parse_body(body) if body else {}
        if req is None or not isinstance(req, dict):
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        conversation_id = req.get("id") or req.get("conversation_id") or f"conv_{uuid.uuid4().hex}"
        conversation_id = _validate_conversation_id(conversation_id)
        info = create_conversation(conversation_id, select=req.get("select", True) is not False)
        self.send_json(info, 201, conversation_id)

    def _handle_conversation_select(self, path: str):
        conversation_id = self._conversation_path_id(path, "/select")
        info = select_conversation(conversation_id)
        self.send_json(
            info or {"error": {"message": "conversation not found"}},
            200 if info else 404,
            conversation_id if info else None,
        )

    def _handle_conversation_reset(self, path: str):
        conversation_id = self._conversation_path_id(path, "/reset")
        reset_conversation(conversation_id)
        _clear_chat_history(conversation_id)
        self.send_json({"id": conversation_id, "reset": True}, conversation_id=conversation_id)

    def _handle_conversation_attach(self, path: str, body: bytes):
        conversation_id = self._conversation_path_id(path, "/attach")
        req = self._parse_body(body)
        if req is None or not isinstance(req, dict):
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        metadata = req.get("metadata")
        if metadata is None:
            metadata = [req.get("cid"), req.get("rid"), req.get("rcid")]
        if (not isinstance(metadata, list) or len(metadata) < 3
                or not all(isinstance(value, str) and value for value in metadata[:3])):
            self.send_json({"error": {"message": "attach requires non-empty cid, rid, and rcid"}}, 400)
            return
        info = create_conversation(
            conversation_id, metadata, select=req.get("select", True) is not False
        )
        _clear_chat_history(conversation_id)
        self.send_json(info, conversation_id=conversation_id)

    # ─── /v1/chat/completions ─────────────────────────────────────────────────

    def _handle_chat(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        requested_model = req.get("model")
        if not isinstance(requested_model, str) or not requested_model:
            self.send_json({"error": {"message": "model is required"}}, 400)
            return
        model_name, attempts, err = resolve_model_chain(requested_model)
        combo = requested_model in configured_model_combos()
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        try:
            conversation_id = self._request_conversation_id(req)
        except ValueError as error:
            self.send_json({"error": {"message": str(error)}}, 400)
            return
        tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        messages = req.get("messages", [])
        incremental_messages = _incremental_chat_messages(messages, conversation_id)
        prompt, images = messages_to_prompt(incremental_messages, tools, tool_choice)
        log(
            f"Chat request: model={requested_model} conversation={conversation_id or 'default'} "
            f"messages={len(messages) if isinstance(messages, list) else 0} "
            f"images={len(images)} stream={bool(req.get('stream', False))}"
        )
        if not prompt.strip():
            self.send_json({"error": {"message": "empty prompt"}}, 400)
            return

        stream = req.get("stream", False)
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        try:
            file_refs = _upload_images(images)
        except ImageAuthenticationRequired as e:
            self.send_json({"error": {"message": str(e)}}, 400)
            return
        except RuntimeError as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return

        if stream and (not tools or tool_choice == "none"):
            try:
                self._start_sse(conversation_id)
                first_chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                self._write_sse(f"data: {json.dumps(first_chunk)}\n\n")
                for delta in _generate_stream_attempts(prompt, attempts, file_refs, combo, conversation_id):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                    self._write_sse(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
                end = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                       "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self._write_sse(f"data: {json.dumps(end)}\n\n")
                self._write_sse(b"data: [DONE]\n\n")
                self._finish_sse()
                _commit_chat_messages(messages, conversation_id)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Stream error: {e}")
            return

        try:
            text = _generate_attempts(prompt, attempts, file_refs, combo, conversation_id)
        except Exception as e:
            message, status = _upstream_error_response(e, bool(images))
            self.send_json({"error": {"message": message}}, status)
            return

        _commit_chat_messages(messages, conversation_id)
        tool_calls = None
        if tools and text and tool_choice != "none":
            text, tool_calls = parse_tool_calls(text)
        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"

        if stream:
            self._start_sse(conversation_id)
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": model_name, "choices": [{"index": 0, "delta": msg, "finish_reason": finish}]}
            self._write_sse(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
            self._write_sse(b"data: [DONE]\n\n")
            self._finish_sse()
        else:
            self.send_json({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model_name, "conversation_id": conversation_id,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt)//4, "completion_tokens": len(text or "")//4,
                          "total_tokens": (len(prompt)+len(text or ""))//4},
            }, conversation_id=conversation_id)

    # ─── /v1/responses (Codex CLI) ───────────────────────────────────────────

    def _handle_responses(self, body: bytes):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        requested_model = req.get("model")
        if not isinstance(requested_model, str) or not requested_model:
            self.send_json({"error": {"message": "model is required"}}, 400)
            return
        model_name, attempts, err = resolve_model_chain(requested_model)
        combo = requested_model in configured_model_combos()
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        try:
            conversation_id = self._request_conversation_id(req)
        except ValueError as error:
            self.send_json({"error": {"message": str(error)}}, 400)
            return
        input_items = req.get("input", [])
        tools = req.get("tools")
        messages = []
        if req.get("instructions"):
            messages.append({"role": "system", "content": req["instructions"]})
        if isinstance(input_items, str):
            messages.append({"role": "user", "content": input_items})
        elif isinstance(input_items, list):
            for item in input_items:
                if isinstance(item, str):
                    messages.append({"role": "user", "content": item})
                elif isinstance(item, dict):
                    if item.get("type") == "function_call_output":
                        messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""),
                                         "name": item.get("name", ""), "content": item.get("output", "")})
                    elif item.get("type") in ("input_text", "input_image", "image"):
                        messages.append({"role": "user", "content": [item]})
                    elif item.get("role") == "assistant" or (item.get("type") == "message" and item.get("role") == "assistant"):
                        cp = item.get("content", [])
                        text_acc, tc_list = "", []
                        if isinstance(cp, list):
                            for c in cp:
                                if isinstance(c, dict):
                                    if c.get("type") == "output_text":
                                        text_acc += c.get("text", "")
                                    elif c.get("type") == "function_call":
                                        tc_list.append(c)
                        elif isinstance(cp, str):
                            text_acc = cp
                        m = {"role": "assistant", "content": text_acc or None}
                        if tc_list:
                            m["tool_calls"] = [{"id": tc.get("call_id", f"call_{i}"), "type": "function",
                                                "function": {"name": tc.get("name",""), "arguments": tc.get("arguments","{}")}}
                                               for i, tc in enumerate(tc_list)]
                        messages.append(m)
                    else:
                        role = item.get("role", "user")
                        messages.append({"role": role, "content": item.get("content", "")})

        if tools:
            tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
                     if t.get("type") == "function" and "function" not in t else t for t in tools]

        tool_choice = req.get("tool_choice", "auto")
        prompt, images = messages_to_prompt(messages, tools, tool_choice)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty input"}}, 400)
            return

        try:
            file_refs = _upload_images(images)
            text = _generate_attempts(prompt, attempts, file_refs, combo, conversation_id)
        except ImageAuthenticationRequired as e:
            self.send_json({"error": {"message": str(e)}}, 400)
            return
        except Exception as e:
            message, status = _upstream_error_response(e, bool(images))
            self.send_json({"error": {"message": message}}, status)
            return

        tool_calls = None
        if tools and text and tool_choice != "none":
            text, tool_calls = parse_tool_calls(text)

        rid = f"resp_{uuid.uuid4().hex[:16]}"
        mid = f"msg_{uuid.uuid4().hex[:12]}"
        output = []
        if tool_calls:
            for tc in tool_calls:
                output.append({"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                               "name": tc["function"]["name"], "arguments": tc["function"]["arguments"], "status": "completed"})
        if text or not tool_calls:
            output.append({"type": "message", "id": mid, "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": text or "", "annotations": []}]})

        if req.get("stream"):
            self._start_sse(conversation_id)
            sequence_number = 0

            def emit(event_type, **fields):
                nonlocal sequence_number
                sequence_number += 1
                event = {
                    "type": event_type,
                    "sequence_number": sequence_number,
                    **fields,
                }
                self._write_sse(
                    f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
                )

            usage = {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(text or "") // 4,
                "total_tokens": (len(prompt) + len(text or "")) // 4,
            }
            base_response = {
                "id": rid,
                "object": "response",
                "created_at": int(time.time()),
                "model": model_name,
            }
            emit(
                "response.created",
                response={
                    **base_response,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            )
            emit(
                "response.in_progress",
                response={
                    **base_response,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            )
            for output_index, item in enumerate(output):
                if item["type"] == "function_call":
                    pending_item = {
                        "type": "function_call",
                        "id": item["id"],
                        "call_id": item["call_id"],
                        "name": item["name"],
                        "arguments": "",
                        "status": "in_progress",
                    }
                    emit(
                        "response.output_item.added",
                        output_index=output_index,
                        item=pending_item,
                    )
                    emit(
                        "response.function_call_arguments.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        delta=item["arguments"],
                    )
                    emit(
                        "response.function_call_arguments.done",
                        item_id=item["id"],
                        output_index=output_index,
                        arguments=item["arguments"],
                    )
                    emit(
                        "response.output_item.done",
                        output_index=output_index,
                        item=item,
                    )
                elif item["type"] == "message":
                    pending_item = {
                        "type": "message",
                        "id": item["id"],
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    }
                    emit(
                        "response.output_item.added",
                        output_index=output_index,
                        item=pending_item,
                    )
                    for content_index, content_part in enumerate(item["content"]):
                        event_fields = {
                            "item_id": item["id"],
                            "output_index": output_index,
                            "content_index": content_index,
                        }
                        emit(
                            "response.content_part.added",
                            **event_fields,
                            part={
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        )
                        emit(
                            "response.output_text.delta",
                            **event_fields,
                            delta=content_part["text"],
                        )
                        emit(
                            "response.output_text.done",
                            **event_fields,
                            text=content_part["text"],
                        )
                        emit(
                            "response.content_part.done",
                            **event_fields,
                            part=content_part,
                        )
                    emit(
                        "response.output_item.done",
                        output_index=output_index,
                        item=item,
                    )
            emit(
                "response.completed",
                response={
                    **base_response,
                    "status": "completed",
                    "output": output,
                    "usage": usage,
                },
            )
            self._finish_sse()
        else:
            self.send_json({"id": rid, "object": "response", "created_at": int(time.time()), "status": "completed",
                            "model": model_name, "conversation_id": conversation_id, "output": output,
                            "usage": {"input_tokens": len(prompt)//4, "output_tokens": len(text or "")//4, "total_tokens": (len(prompt)+len(text or ""))//4}},
                           conversation_id=conversation_id)

    # ─── /v1beta/models (Google Gemini CLI) ──────────────────────────────────

    def _handle_google_generate(self, body: bytes, stream: bool):
        req = self._parse_body(body)
        if req is None:
            self.send_json({"error": {"message": "invalid JSON"}}, 400)
            return
        m = re.match(r'/v1beta/models/([^:?]+)', self.path)
        if not m:
            self.send_json({"error": {"message": "model is required in URL"}}, 400)
            return
        requested_model = m.group(1)
        model_name, attempts, err = resolve_model_chain(requested_model)
        combo = requested_model in configured_model_combos()
        if err:
            self.send_json({"error": {"message": err}}, 400)
            return

        try:
            conversation_id = self._request_conversation_id(req)
        except ValueError as error:
            self.send_json({"error": {"message": str(error)}}, 400)
            return
        tool_config = req.get("toolConfig", {})
        fc_mode = tool_config.get("functionCallingConfig", {}).get("mode", "AUTO")
        has_tools = bool(req.get("tools")) and fc_mode != "NONE"
        prompt, images = google_contents_to_prompt(req)
        if not prompt.strip():
            self.send_json({"error": {"message": "empty content"}}, 400)
            return

        try:
            file_refs = _upload_images(images)
        except ImageAuthenticationRequired as e:
            self.send_json({"error": {"message": str(e)}}, 400)
            return
        except RuntimeError as e:
            self.send_json({"error": {"message": f"upstream error: {e}"}}, 502)
            return
        log(f"Google API: model={model_name} stream={stream} tools={has_tools} prompt_len={len(prompt)}")

        if stream and not has_tools:
            try:
                self._start_sse(conversation_id)
                full_text = ""
                for delta in _generate_stream_attempts(prompt, attempts, file_refs, combo, conversation_id):
                    if not delta:
                        continue
                    full_text += delta
                    chunk_obj = {
                        "candidates": [{"content": {"parts": [{"text": delta}], "role": "model"}, "index": 0}],
                        "modelVersion": model_name,
                    }
                    self._write_sse(f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n")
                final_chunk = {
                    "candidates": [{"finishReason": "STOP", "index": 0}],
                    "usageMetadata": {
                        "promptTokenCount": len(prompt) // 4,
                        "candidatesTokenCount": len(full_text) // 4,
                        "totalTokenCount": (len(prompt) + len(full_text)) // 4,
                    },
                    "modelVersion": model_name,
                }
                self._write_sse(f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n")
                self._finish_sse()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Google stream error: {e}")
            return

        try:
            text = _generate_attempts(prompt, attempts, file_refs, combo, conversation_id)
        except Exception as e:
            message, status = _upstream_error_response(e, bool(images))
            self.send_json({"error": {"message": message}}, status)
            return

        if not text:
            log("Warning: empty response from Gemini")

        response_parts = []
        if has_tools and text:
            clean_text, function_calls = parse_google_function_calls(text)
            if function_calls:
                if clean_text:
                    response_parts.append({"text": clean_text})
                for fc in function_calls:
                    response_parts.append({"functionCall": {"name": fc["name"], "args": fc["args"]}})
            else:
                response_parts.append({"text": text})
        else:
            response_parts.append({"text": text or "I apologize, but I was unable to generate a response. Please try again."})

        candidate = {
            "content": {"parts": response_parts, "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }
        usage = {
            "promptTokenCount": len(prompt) // 4,
            "candidatesTokenCount": len(text or "") // 4,
            "totalTokenCount": (len(prompt) + len(text or "")) // 4,
        }
        response_obj = {
            "candidates": [candidate],
            "usageMetadata": usage,
            "modelVersion": model_name,
        }

        if stream:
            self._start_sse(conversation_id)
            self._write_sse(f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n")
            self._finish_sse()
        else:
            response_obj["conversationId"] = conversation_id
            self.send_json(response_obj, conversation_id=conversation_id)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
