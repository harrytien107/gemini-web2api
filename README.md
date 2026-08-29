# gemini-web2api

<p align="center">
  <img src="logo.png" width="200" alt="gemini-web2api logo">
</p>

[中文文档](README_CN.md)

Convert Google Gemini's web interface into an OpenAI-compatible API. Zero cost, cross-platform, single file.

## Features

- **Optional API Keys**: no auth when `api_keys` is empty, OpenAI-style Bearer auth when configured
- **OpenAI Compatible**: Drop-in replacement for `/v1/chat/completions` and `/v1/models`
- **Tool Calling**: Full function calling support (OpenAI format)
- **Multiple Models**: Flash (3.6), Extended Thinking (20k+ char output), Pro, Auto, Lite
- **Thinking Depth**: Adjustable via `@think=N` suffix (0=deepest, 4=shallowest)
- **Web Search**: Built-in internet access (Gemini's native search)
- **Cross-Platform**: Pure Python, single optional dependency (`httpx` for streaming)
- **Streaming**: SSE streaming support via `httpx`
- **Codex CLI**: Responses API (`/v1/responses`) for OpenAI Codex integration
- **Gemini CLI**: Google native API (`/v1beta/models`) for Gemini CLI compatibility

## Quick Start

```bash
pip install httpx
python gemini_web2api.py
```

Server starts at `http://localhost:8081/v1`.

### Build a portable Windows executable

Run the build on Windows; PyInstaller executables are specific to the OS that builds them. Python 3.8 or newer is required for the build only.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-windows-exe.ps1
```

The script creates an isolated `.venv-build`, installs the streaming dependency and PyInstaller, runs the test suite, and writes two tray executables:

```text
dist\gemini-web2api.exe
dist\gemini-web2api-cookie.exe
```

Run them without Python or a virtual environment. They start without a console window and remain available from the Windows notification area with the Gemini icon. Double-click the tray icon to open the conversation manager; right-click it to see the live **Auth: Cookie loaded** or **Auth: Anonymous** status and actions for **Manage conversations**, **Open API**, **Copy endpoint**, **Open config.json**, and **Exit**. The status refreshes whenever the menu opens, so auth-file hot reloads are reflected immediately. **Copy endpoint** copies `http://localhost:8081/v1` using the active port. Use **Exit** to stop the HTTP server cleanly:

```powershell
.\dist\gemini-web2api.exe
.\dist\gemini-web2api.exe --port 8082
.\dist\gemini-web2api.exe --config .\config.json --cookie-file .\gemini-auth.json
```

On first start, both executables create `config.json` beside themselves if it is missing. `gemini-web2api-cookie.exe` also creates an empty `gemini-auth.json` template. Existing files are never overwritten. The cookie executable then automatically loads both sibling files:

```text
dist\
  gemini-web2api-cookie.exe
  config.json
  gemini-auth.json
```

```powershell
.\dist\gemini-web2api-cookie.exe
```

The generated auth template has an empty `cookie`; this is valid and runs anonymously until replaced by an extension export. Exporting a newer `gemini-auth.json` over the running file hot-reloads cookies, `auth_user`, XSRF, and Gemini build metadata before the next request; restarting the EXE is unnecessary. Explicit `--config` and `--cookie-file` arguments override sibling files. A configured `cookie_file` takes precedence over sibling auth; when neither supplies credentials, the cookie EXE still accepts a sibling legacy `cookie.txt`.

When running from Python without tray mode, startup prints `Cookie: yes (<path>)` only after valid auth data is loaded, `Cookie: error (<path>): <reason>` for invalid auth JSON, `Cookie: missing (<path>)` for a configured missing path, `Cookie: none (anonymous; empty template: <path>)` for the generated empty template, or `Cookie: none (anonymous)` when no path is configured.

Keep `config.json` and auth files external to the executables so they can be changed without rebuilding. Never distribute the auth file with the executable. The build environment and generated `build`/`dist` directories are ignored by Git.

## Client Configuration

### Cherry Studio / ChatBox / any OpenAI client

| Field | Value |
|-------|-------|
| Base URL | `http://localhost:8081/v1` |
| API Key | any `api_keys` value from `config.json`; anything if not configured |
| Model | `gemini-3.5-flash-thinking` |

### curl

#### bash / macOS / Linux

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hello!"}]}'
```

#### PowerShell (Windows)

```powershell
curl.exe --% http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer sk-your-key" -d "{\"model\":\"gemini-3.5-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
```

> Note: On Windows PowerShell, use `curl.exe` and `--%` so PowerShell does not reinterpret JSON quoting or curl options.

### OpenAI Python SDK

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-your-key")
resp = client.chat.completions.create(
    model="gemini-3.5-flash-thinking",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(resp.choices[0].message.content)
```

### Gemini CLI

```bash
export GEMINI_API_KEY=none
export GOOGLE_GEMINI_BASE_URL=http://localhost:8081
gemini
```

Supports Google native API endpoints:
- `GET /v1beta/models` — list models
- `POST /v1beta/models/{model}:generateContent` — non-streaming
- `POST /v1beta/models/{model}:streamGenerateContent` — streaming (SSE)

## Persistent conversations

For portable Windows builds, double-click the tray icon or right-click it and choose **Manage conversations**. The browser page creates, selects, lists, and resets sessions. If `api_keys` is `[]`, leave its API-key field empty. No command line is required.

Create and select a conversation once, then use an ordinary OpenAI client without custom headers. The selected conversation remains active across model changes and executable restarts:

```powershell
curl.exe --% -X POST http://127.0.0.1:8081/v1/conversations -H "Content-Type: application/json" -H "Authorization: Bearer sk-your-key" -d "{\"id\":\"my-overlay-chat\"}"
```

Clients that support custom headers can select a conversation per request instead of changing the process-wide active conversation:

```http
X-Conversation-ID: my-overlay-chat
```

The JSON request field `conversation_id` is also accepted. Selection precedence is header, request field, then the process-wide active conversation.

Conversation management endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /v1/conversations` | Create and select a conversation. Body: `{"id":"my-chat"}`. Omit `id` to generate one. |
| `GET /v1/conversations` | List saved conversations for the current Gemini account. |
| `GET /v1/conversations/{id}` | Inspect saved Gemini metadata. |
| `POST /v1/conversations/{id}/select` | Make a saved conversation the default for clients without custom headers. |
| `POST /v1/conversations/{id}/reset` | Delete its metadata and clear it if selected. |
| `POST /v1/conversations/{id}/attach` | Attach an existing Gemini Web chat using `cid`, `rid`, and `rcid`. |

Attach an existing Gemini Web conversation after obtaining all three private metadata values:

```powershell
curl.exe --% -X POST http://127.0.0.1:8081/v1/conversations/site-chat/attach -H "Content-Type: application/json" -H "Authorization: Bearer sk-your-key" -d "{\"cid\":\"...\",\"rid\":\"...\",\"rcid\":\"...\"}"
```

The Gemini URL generally exposes only `cid`; continuing the latest branch reliably also requires `rid` and `rcid`. The adapter does not currently discover those two values from a URL. Conversations created through this API capture all three automatically.

Named metadata is stored in `gemini-conversations.json` beside a portable executable. Set `conversation_store_file` to override that location. The store contains private Gemini conversation IDs and an account-session fingerprint, not raw cookies, but it should still remain private. A selected conversation deliberately survives rebuilt message history; use its reset endpoint when the overlay starts a genuinely new chat.

## Available Models

| Model | Description | Output |
|-------|-------------|--------|
| `gemini-3.6-flash` | All-around model (latest) | ~12k chars |
| `gemini-3.5-flash` | Alias for gemini-3.6-flash | ~12k chars |
| `gemini-3.5-flash-thinking` | Extended thinking, longest output | **~20k chars** |
| `gemini-3.5-flash-thinking-lite` | Adaptive thinking depth | ~15k chars |
| `gemini-3.1-pro` | Advanced math & code (needs cookie) | ~12k chars |
| `gemini-auto` | Auto model selection | varies |
| `gemini-flash-lite` | Fastest answers, lightweight | ~10k chars |

### Thinking Depth

Append `@think=N` to any model name:

```
gemini-3.5-flash-thinking@think=0   # deepest (default)
gemini-3.5-flash-thinking@think=2   # medium
gemini-3.5-flash-thinking@think=4   # shallowest
```

## Optional: Cookie for Pro

Anonymous access works for all models, but `gemini-3.1-pro` routes to Flash without authentication. To get real Pro routing, you need a **Gemini Advanced (paid subscription)** account session.

### Recommended: Gemini Cookie Sync extension

1. Open `chrome://extensions`, enable Developer mode, and click **Load unpacked**. Click **Reload** there after updating an already-loaded extension.
2. Select the `gemini-cookie-sync-extension` folder.
3. Open `https://gemini.google.com/app`, sign in, and refresh the page.
4. Open the extension, click **Inspect session**, then **Export gemini-auth.json**.
5. The extension downloads exactly `gemini-auth.json` and overwrites the previous download instead of using a blob UUID. Move it beside `gemini-web2api-cookie.exe`; the running EXE hot-reloads it.

The extension exports the cookie string, `SAPISID`, account index, XSRF token, and current Gemini build identifier. The server validates and hot-reloads this file when it changes. If a replacement is briefly incomplete or invalid, active requests continue using the last valid snapshot.

Precedence is: explicit CLI path > sibling `gemini-auth.json` > `cookie_file` from `config.json` > anonymous/default configuration. Auth metadata present in `gemini-auth.json` overrides matching `auth_user`, `xsrf_token`, and `gemini_bl` values from `config.json`; missing metadata falls back to config.

Manual `cookie.txt` and JSON files containing at least `cookie` plus optional `sapisid` remain supported:

```bash
python gemini_web2api.py --cookie-file gemini-auth.json
```

If authenticated requests return HTTP 400 with an XSRF error, refresh Gemini Web and export `gemini-auth.json` again.

Pro routing requires **Gemini Advanced** (paid subscription). A free Google account cookie will authenticate but silently fall back to Flash.

## Configuration

Create `config.json` in the same directory:

```json
{
  "port": 8081,
  "host": "0.0.0.0",
  "retry_attempts": 3,
  "retry_delay_sec": 2,
  "request_timeout_sec": 180,
  "gemini_bl": "boq_assistant-bard-web-server_20260716.08_p0",
  "auth_user": null,
  "xsrf_token": null,
  "model_combos": {
    "gemini-combo": {
      "strategy": "fallback",
      "models": [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-thinking",
        "gemini-3.5-flash-thinking-lite"
      ]
    },
    "gemini-deep": {
      "strategy": "round_robin",
      "models": [
        "gemini-3.7-flash-thinking@think=0",
        "gemini-3.6-flash-thinking@think=0",
        "gemini-3.5-flash-thinking@think=0"
      ]
    }
  },
  "api_keys": ["sk-your-key"],
  "cookie_file": null,
  "proxy": null,
  "log_requests": true,
  "temporary_chats": false,
  "conversation_store_file": null
}
```

Set `temporary_chats` to `true` to use Gemini Web temporary chats instead of
persisting conversations to the account history. Named conversation selection is intended for persistent chats; temporary mode may cause Gemini to clear upstream context.

Each `model_combos` key becomes a virtual model exposed by `/v1/models`; clients can request `gemini-combo`, `gemini-deep`, or any other configured name. `fallback` starts every request at the first model. `round_robin` rotates the first model per request, then uses the remaining models as fallbacks. Rotation is in memory and resets when the process restarts. An array value is shorthand for `fallback`. OpenAI-compatible Chat Completions and Responses requests must include a non-empty `model`; missing models return HTTP 400. Models fall back after an upstream error or empty response. `@think=N` overrides the thinking level for that attempt. Streaming falls back only before any content is emitted, preventing mixed model output. Combo nesting, invalid strategies, and invalid model names reject the request.

When `api_keys` is `[]`, authentication is disabled. When one or more keys are set, `/v1/*` endpoints require `Authorization: Bearer <key>` or `x-api-key: <key>`.

## Docker

```bash
cp config.example.json config.json
docker build -t gemini-web2api .
docker run -d --name gemini-web2api -p 8081:8081 -v ./config.json:/app/config.json gemini-web2api
```

Or use Docker Compose:

```bash
cp config.example.json config.json
docker compose up -d
```

To mount a cookie file:

```bash
docker run -d --name gemini-web2api -p 8081:8081 -v ./config.json:/app/config.json -v ./cookie.txt:/app/cookie.txt gemini-web2api
```

Set `"cookie_file": "/app/cookie.txt"` in `config.json`.

> **Note**: If you get empty responses (`content: null`) with Docker's default bridge network, switch to host networking: `docker run --network host ...` or add `network_mode: host` in your compose file. This is caused by Gemini's upstream rejecting requests from certain Docker NAT IP ranges.

## Proxy

If you cannot access `gemini.google.com` directly (connection timeout), configure a proxy:

**Method 1: CLI argument**
```bash
python gemini_web2api.py --proxy http://127.0.0.1:7890
```

**Method 2: config.json**
```json
{"proxy": "http://127.0.0.1:7890"}
```

**Method 3: Environment variable** (auto-detected)
```bash
export HTTPS_PROXY=http://127.0.0.1:7890
python gemini_web2api.py
```

Works with Clash, V2Ray, Shadowsocks, or any HTTP proxy.

## Tool Calling

```python
resp = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }]
)
```

## Image Input

OpenAI-style multimodal messages are supported for Chat Completions and the
Responses API. Use either HTTP(S) image URLs or base64 data URLs:

```python
resp = client.chat.completions.create(
    model="gemini-3.6-flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
        ]
    }]
)
```

## Limitations

- **Image upload may require cookies**: Multimodal input uses Gemini Web's image upload endpoint. If anonymous upload fails, configure a Gemini cookie.
- **Not real Pro/Ultra**: Without a paid subscription cookie, `gemini-3.1-pro` routes to the same Flash model. The "Pro" label is a UI preference, not a backend model switch.
- **One active conversation**: Sequential overlay requests reuse one Gemini Web conversation. Clearing or replacing the overlay message history resets the upstream conversation. Run separate server instances on separate ports for concurrent independent overlays.
- **Rate limits**: Google may throttle high-frequency requests. The server retries automatically but sustained heavy use may be blocked.

## Requirements

- Python 3.8+
- `httpx` (`pip install httpx`) — used for streaming requests
- Network access to `gemini.google.com` (proxy/VPN may be needed in some regions)

## How It Works

This tool reverse-engineers Google Gemini's web StreamGenerate protocol. It sends requests to the same endpoint that the Gemini web app uses, converting between OpenAI's API format and Gemini's internal protobuf-like format.

The model selection is controlled by field `[79]` in the request payload, mapped from Gemini's frontend JavaScript source (`MODE_CATEGORY` enum).

## Acknowledgments

- Inspired by the open-source API proxy ecosystem

## License

MIT

---

## 致谢

本项目的开发 agent 能力由 [GenericAgent](https://github.com/lsdefine/GenericAgent) 提供。

### 🚩 友情链接

[![GenericAgent](https://img.shields.io/badge/Agent_Framework-GenericAgent-orange?style=for-the-badge&logo=github)](https://github.com/lsdefine/GenericAgent)
[![LinuxDo](https://img.shields.io/badge/社区-LinuxDo-blue?style=for-the-badge)](https://linux.do/)
