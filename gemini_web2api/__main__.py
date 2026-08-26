"""Entry point: python -m gemini_web2api"""
import argparse
import os
import sys

from .config import CONFIG, load_config, find_config
from .models import MODELS
from .gemini import HAS_HTTPX, auth_status, log
from .server import GeminiHandler, ThreadedServer
from . import __version__


def main():
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    parser = argparse.ArgumentParser(description="Gemini Web to OpenAI API")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--cookie-file", type=str, default=None)
    parser.add_argument("--proxy", type=str, default=None, help="HTTP proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--tray", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version=f"gemini-web2api {__version__}")
    args = parser.parse_args()

    config_path = args.config or os.environ.get("GEMINI_WEB2API_CONFIG") or find_config()
    if config_path:
        load_config(config_path)

    if args.port:
        CONFIG["port"] = args.port
    if args.cookie_file:
        CONFIG["cookie_file"] = args.cookie_file
    if args.proxy:
        CONFIG["proxy"] = args.proxy

    port = CONFIG["port"]
    status = auth_status()
    executable = os.path.basename(sys.executable) if getattr(sys, "frozen", False) else "python"
    log(
        f"Startup: executable={executable} port={port} "
        f"auth={'authenticated' if status['loaded'] else 'anonymous'}"
    )
    if status["loaded"]:
        cookie_status = f"yes ({status['path']})"
    elif status["error"]:
        cookie_status = f"error ({status['path']}): {status['error']}"
    elif status["exists"]:
        cookie_status = f"none (anonymous; empty template: {status['path']})"
    elif status["path"]:
        cookie_status = f"missing ({status['path']})"
    else:
        cookie_status = "none (anonymous)"
    server = ThreadedServer((CONFIG["host"], port), GeminiHandler)
    if not args.tray:
        print(f"gemini-web2api v{__version__}")
        print(f"  Listening: http://0.0.0.0:{port}")
        print(f"  Base URL:  http://localhost:{port}/v1")
        print(f"  Models:    {', '.join(MODELS.keys())}")
        print(f"  Cookie:    {cookie_status}")
        print(f"  Proxy:     {CONFIG.get('proxy') or 'system env'}")
        print(f"  Streaming: {'httpx (true streaming)' if HAS_HTTPX else 'urllib (buffered)'}")
        print(f"  Temporary: {'yes' if CONFIG.get('temporary_chats', False) else 'no'}")
        print(flush=True)
    if args.tray and sys.platform == "win32":
        from .tray import run_tray
        run_tray(server, config_path)
        return
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
