"""HTTP server: OpenAI-compatible API endpoints."""
import hashlib
import json
import time
import uuid
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from .config import CONFIG
from .models import MODELS, configured_model_combos, resolve_model_chain
from .gemini import GeminiUpstreamError, generate, generate_stream, log, refresh_auth, reset_conversation
from .tools import messages_to_prompt, parse_tool_calls, google_contents_to_prompt, parse_google_function_calls
from .multimodal import detect_image_mime, fetch_image_bytes, upload_image
from . import __version__


_chat_history_lock = threading.Lock()
_chat_history = []


def _incremental_chat_messages(messages: list) -> list:
    """Return only messages Gemini has not already seen, resetting on a new overlay chat."""
    if not isinstance(messages, list):
        return []
    with _chat_history_lock:
        previous = list(_chat_history)
    continues = bool(previous) and len(messages) >= len(previous) and messages[:len(previous)] == previous
    if previous and not continues:
        reset_conversation()
    if not continues:
        return messages
    new_messages = messages[len(previous):]
    # ponytail: one upstream session serves one overlay; add a client conversation ID if multiplexing clients.
    return [message for message in new_messages if message.get("role", "user") != "assistant"]


def _commit_chat_messages(messages: list) -> None:
    global _chat_history
    with _chat_history_lock:
        _chat_history = list(messages)


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


def _generate_attempts(prompt, attempts, file_refs, combo):
    errors = []
    terminal_errors = []
    for name, model_id, think_mode, extra_fields in attempts:
        log(f"Model {name}: calling")
        caught = None
        try:
            text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
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


def _generate_stream_attempts(prompt, attempts, file_refs, combo):
    errors = []
    terminal_errors = []
    for name, model_id, think_mode, extra_fields in attempts:
        emitted = False
        caught = None
        log(f"Model {name}: calling")
        try:
            for delta in generate_stream(prompt, model_id, think_mode, file_refs, extra_fields):
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

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
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
            if self.path == "/v1/models":
                self.send_json({"object": "list", "data": [
                    {"id": n, "object": "model", "created": 1700000000,
                     "owned_by": "google", "description": c["desc"]}
                    for n, c in _model_catalog().items()
                ]})
            elif self.path.startswith("/v1beta/models"):
                self.send_json({"models": [
                    {"name": f"models/{n}", "displayName": n, "description": c["desc"],
                     "supportedGenerationMethods": ["generateContent", "streamGenerateContent"]}
                    for n, c in _model_catalog().items()
                ]})
            elif self.path == "/":
                self.send_json({"status": "ok", "version": __version__, "models": list(MODELS.keys())})
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            refresh_auth()
            if self.path.startswith("/v1") and not self._authorized():
                self.send_json({"error": {"message": "invalid api key"}}, 401)
                return
            body = self._read_request_body()
            if self.path == "/v1/chat/completions":
                self._handle_chat(body)
            elif self.path == "/v1/responses":
                self._handle_responses(body)
            elif ":streamGenerateContent" in self.path:
                self._handle_google_generate(body, stream=True)
            elif ":generateContent" in self.path:
                self._handle_google_generate(body, stream=False)
            else:
                self.send_json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log(f"POST error: {e}")
            try:
                self.send_json({"error": {"message": str(e)}}, 500)
            except:
                pass

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

        tools = req.get("tools")
        tool_choice = req.get("tool_choice", "auto")
        messages = req.get("messages", [])
        incremental_messages = _incremental_chat_messages(messages)
        prompt, images = messages_to_prompt(incremental_messages, tools, tool_choice)
        log(
            f"Chat request: model={requested_model} messages={len(messages) if isinstance(messages, list) else 0} "
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
                self._start_sse()
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
                for delta in _generate_stream_attempts(prompt, attempts, file_refs, combo):
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": model_name, "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]}
                    self._write_sse(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
                end = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                       "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                self._write_sse(f"data: {json.dumps(end)}\n\n")
                self._write_sse(b"data: [DONE]\n\n")
                self._finish_sse()
                _commit_chat_messages(messages)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log(f"Stream error: {e}")
            return

        try:
            text = _generate_attempts(prompt, attempts, file_refs, combo)
        except Exception as e:
            message, status = _upstream_error_response(e, bool(images))
            self.send_json({"error": {"message": message}}, status)
            return

        _commit_chat_messages(messages)
        tool_calls = None
        if tools and text and tool_choice != "none":
            text, tool_calls = parse_tool_calls(text)
        msg = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        finish = "tool_calls" if tool_calls else "stop"

        if stream:
            self._start_sse()
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": model_name, "choices": [{"index": 0, "delta": msg, "finish_reason": finish}]}
            self._write_sse(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
            self._write_sse(b"data: [DONE]\n\n")
            self._finish_sse()
        else:
            self.send_json({
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": len(prompt)//4, "completion_tokens": len(text or "")//4,
                          "total_tokens": (len(prompt)+len(text or ""))//4},
            })

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
            text = _generate_attempts(prompt, attempts, file_refs, combo)
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
            self._start_sse()
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
                            "model": model_name, "output": output,
                            "usage": {"input_tokens": len(prompt)//4, "output_tokens": len(text or "")//4, "total_tokens": (len(prompt)+len(text or ""))//4}})

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
                self._start_sse()
                full_text = ""
                for delta in _generate_stream_attempts(prompt, attempts, file_refs, combo):
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
            text = _generate_attempts(prompt, attempts, file_refs, combo)
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
            self._start_sse()
            self._write_sse(f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n")
            self._finish_sse()
        else:
            self.send_json(response_obj)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
