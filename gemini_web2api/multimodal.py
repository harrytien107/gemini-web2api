"""Multimodal file upload for Gemini image input."""
import hashlib
import http.cookiejar
import re
import secrets
import time
import urllib.request
from urllib.parse import urlparse

from .config import CONFIG
from .gemini import (
    _account_prefix,
    _get_ssl_ctx,
    log,
    make_sapisidhash,
    refresh_auth,
    set_runtime_page_metadata,
)


def _auth_headers(auth: dict) -> dict:
    account_prefix = _account_prefix(auth)
    headers = {
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(auth["auth_user"])
    if auth["cookie"]:
        headers["Cookie"] = auth["cookie"]
    if auth["sapisid"]:
        headers["Authorization"] = make_sapisidhash(auth["sapisid"])
    return headers


_upload_cookie_jar = http.cookiejar.CookieJar()
_upload_opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(_upload_cookie_jar),
    urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
)


def _get_page_tokens(auth: dict = None) -> dict:
    """Fetch account-scoped upload tokens from the Gemini page."""
    auth = auth or refresh_auth()
    account_prefix = _account_prefix(auth)
    headers = _auth_headers(auth)
    try:
        req = urllib.request.Request(
            f"https://gemini.google.com{account_prefix}/app",
            headers=headers,
        )
        proxy = CONFIG.get("proxy")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPCookieProcessor(_upload_cookie_jar),
                urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
            )
            resp = opener.open(req, timeout=30)
        else:
            resp = _upload_opener.open(req, timeout=30)
        html = resp.read().decode()
        tokens = {}
        for key, patterns in [
            ("push_id", (r'"qKIAYe"\s*:\s*"([^"]+)"',)),
            ("pctx", (r'"Ylro7b"\s*:\s*"([^"]+)"',)),
            ("at", (
                r'"SNlM0e"\s*:\s*"([^"]+)"',
                r'"thykhd"\s*:\s*"([^"]+)"',
            )),
            ("bl", (r'"cfb2h"\s*:\s*"([^"]+)"',)),
            ("session_id", (r'"FdrFJe"\s*:\s*"([^"]+)"',)),
        ]:
            for pattern in patterns:
                if m := re.search(pattern, html):
                    tokens[key] = m.group(1)
                    break
        set_runtime_page_metadata(auth, tokens)
        return tokens
    except Exception as e:
        log(f"Page token fetch failed: {e}")
        return {}


_page_tokens_cache = {"tokens": {}, "ts": 0, "auth_key": None}


def _cached_page_tokens(auth: dict = None) -> dict:
    auth = auth or refresh_auth()
    auth_key = (
        str(auth.get("auth_user") or ""),
        hashlib.sha256(auth.get("cookie", "").encode()).digest(),
    )
    now = time.time()
    if auth_key != _page_tokens_cache["auth_key"] or now - _page_tokens_cache["ts"] > 600:
        if auth_key != _page_tokens_cache["auth_key"]:
            _upload_cookie_jar.clear()
        _page_tokens_cache.update({
            "tokens": _get_page_tokens(auth),
            "ts": now,
            "auth_key": auth_key,
        })
    return dict(_page_tokens_cache["tokens"])


def detect_image_mime(image_bytes: bytes, fallback: str = "image/png") -> str:
    """Infer a common raster image MIME type from its file signature."""
    if not isinstance(image_bytes, bytes):
        return fallback
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(image_bytes) >= 12 and image_bytes[4:8] == b"ftyp":
        brand = image_bytes[8:12]
        if brand in (b"avif", b"avis"):
            return "image/avif"
        if brand in (b"heic", b"heix", b"hevc", b"hevx"):
            return "image/heic"
    return fallback


def upload_image(image_bytes: bytes, filename: str = "image.png", mime_type: str = "image/png") -> str:
    """Upload an image as multipart form data and return its Gemini file reference."""
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("image_bytes must be non-empty bytes")
    if not filename or any(char in filename for char in '\r\n"'):
        raise ValueError("filename contains invalid characters")

    auth = refresh_auth()
    tokens = _cached_page_tokens(auth)
    if not tokens.get("push_id"):
        raise RuntimeError("Gemini page omitted upload metadata: push_id")

    boundary = f"----WebKitFormBoundary{secrets.token_hex(12)}"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {mime_type}\r\n\r\n".encode(),
        image_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    headers = {
        **_auth_headers(auth),
        "Push-ID": tokens["push_id"],
        "X-Tenant-Id": "bard-storage",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    req = urllib.request.Request(
        "https://content-push.googleapis.com/upload",
        data=body,
        headers=headers,
        method="POST",
    )
    proxy = CONFIG.get("proxy")
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
            urllib.request.HTTPCookieProcessor(_upload_cookie_jar),
            urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
        )
        resp = opener.open(req, timeout=60)
    else:
        resp = _upload_opener.open(req, timeout=60)

    file_ref = resp.read().decode().strip()
    if not file_ref or not file_ref.startswith("/"):
        raise RuntimeError("Upload returned an invalid file reference")

    log(f"Image uploaded: {filename}")
    return file_ref


def fetch_image_bytes(url: str) -> bytes:
    """Fetch image from URL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        log(f"Image fetch skipped for unsupported URL scheme: {parsed.scheme or 'none'}")
        return b""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        proxy = CONFIG.get("proxy")
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=_get_ssl_ctx()),
            )
            resp = opener.open(req, timeout=30)
        else:
            resp = urllib.request.urlopen(req, context=_get_ssl_ctx(), timeout=30)
        return resp.read()
    except Exception as e:
        log(f"Image fetch failed: {e}")
        return b""
