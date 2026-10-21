"""Protezione minima di /mcp, compatibile con l'OAuth di ChatGPT.

ChatGPT non sa inviare una API key statica. Se MCP_API_KEY è impostata, il
server pubblica i metadati OAuth e una pagina in cui si inserisce quella
chiave una sola volta. Poi ChatGPT conserva il token Bearer emesso dal server.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import os
import secrets
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse

import requests
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

_CODE_TTL_SECONDS = 300
_TOKEN_TTL_SECONDS = 30 * 24 * 3600
_CIMD_TTL_SECONDS = 600
_MAX_FAILURES = 8
_FAILURE_WINDOW_SECONDS = 300

_codes: Dict[str, Dict[str, Any]] = {}
_cimd_cache: Dict[str, tuple[float, Dict[str, Any]]] = {}
_failures: Dict[str, list[float]] = {}
_lock = threading.Lock()


def api_key() -> str:
    return (os.environ.get("MCP_API_KEY") or "").strip()


def auth_enabled() -> bool:
    return bool(api_key())


def _origin() -> str:
    raw = (
        os.environ.get("PUBLIC_URL")
        or os.environ.get("BASE_URL")
        or "https://weaviate-openai-app-sdk-v3.onrender.com"
    )
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return raw.rstrip("/")


def _resource() -> str:
    return _origin() + "/mcp"


def _metadata_url() -> str:
    return _origin() + "/.well-known/oauth-protected-resource/mcp"


def _www_authenticate() -> str:
    return f'Bearer resource_metadata="{_metadata_url()}"'


def _secret_equal(got: str, expected: str) -> bool:
    got_b = got.encode("utf-8")
    expected_b = expected.encode("utf-8")
    if len(got_b) != len(expected_b):
        return False
    return hmac.compare_digest(got_b, expected_b)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def issue_access_token() -> str:
    payload = json.dumps(
        {"exp": int(time.time()) + _TOKEN_TTL_SECONDS, "aud": _resource()},
        separators=(",", ":"),
    ).encode("utf-8")
    body = _b64(payload)
    signature = hmac.new(api_key().encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return body + "." + _b64(signature)


def access_token_valid(token: str) -> bool:
    key = api_key()
    if not key or not token:
        return False
    if _secret_equal(token, key):
        return True
    body, dot, signature = token.partition(".")
    if not dot or not body or not signature:
        return False
    expected = hmac.new(key.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    try:
        got = _b64_decode(signature)
    except Exception:
        return False
    if len(got) != len(expected) or not hmac.compare_digest(got, expected):
        return False
    try:
        payload = json.loads(_b64_decode(body))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("aud") != _resource():
        return False
    exp = payload.get("exp")
    return isinstance(exp, int) and exp > int(time.time())


def _pkce_s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _b64(digest)


def _client_host(request: Request) -> str:
    client = request.client
    return client.host if client else "unknown"


def _too_many_failures(host: str) -> bool:
    now = time.time()
    with _lock:
        recent = [stamp for stamp in _failures.get(host, []) if now - stamp < _FAILURE_WINDOW_SECONDS]
        _failures[host] = recent
        return len(recent) >= _MAX_FAILURES


def _record_failure(host: str) -> None:
    now = time.time()
    with _lock:
        recent = [stamp for stamp in _failures.get(host, []) if now - stamp < _FAILURE_WINDOW_SECONDS]
        recent.append(now)
        _failures[host] = recent


def _chatgpt_client_id(client_id: str) -> bool:
    parsed = urlparse(client_id)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "chatgpt.com"
        and parsed.port in (None, 443)
        and not parsed.username
        and parsed.path.startswith("/oauth/")
        and parsed.path.endswith("/client.json")
        and parsed.query == ""
        and parsed.fragment == ""
    )


def _load_cimd(client_id: str) -> Optional[Dict[str, Any]]:
    if not _chatgpt_client_id(client_id):
        return None
    now = time.time()
    with _lock:
        cached = _cimd_cache.get(client_id)
        if cached and cached[0] > now:
            return cached[1]
    try:
        response = requests.get(client_id, timeout=5, allow_redirects=False)
    except Exception as exc:
        print(f"[mcp-auth] CIMD non raggiungibile: {exc.__class__.__name__}")
        return None
    if response.status_code != 200:
        print(f"[mcp-auth] CIMD HTTP {response.status_code}")
        return None
    try:
        document = response.json()
    except Exception:
        return None
    if not isinstance(document, dict):
        return None
    with _lock:
        _cimd_cache[client_id] = (now + _CIMD_TTL_SECONDS, document)
    return document


def _redirect_allowed(redirect_uri: str, document: Dict[str, Any]) -> bool:
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "https" or parsed.hostname != "chatgpt.com" or parsed.port not in (None, 443):
        return False
    if parsed.username or parsed.query or parsed.fragment:
        return False
    allowed = document.get("redirect_uris")
    return isinstance(allowed, list) and redirect_uri in allowed


def _clean_codes() -> None:
    now = time.time()
    expired = [code for code, data in _codes.items() if data["exp"] <= now]
    for code in expired:
        _codes.pop(code, None)


def _store_code(challenge: str, redirect_uri: str, client_id: str) -> str:
    code = secrets.token_urlsafe(32)
    with _lock:
        _clean_codes()
        _codes[code] = {
            "exp": time.time() + _CODE_TTL_SECONDS,
            "challenge": challenge,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
        }
    return code


def _take_code(code: str) -> Optional[Dict[str, Any]]:
    with _lock:
        _clean_codes()
        data = _codes.pop(code, None)
    if not data or data["exp"] <= time.time():
        return None
    return data


def _oauth_error(status: int, error: str) -> JSONResponse:
    return JSONResponse({"error": error}, status_code=status)


def _form_page(params: Dict[str, str], message: str = "") -> HTMLResponse:
    hidden = []
    for name in (
        "response_type",
        "client_id",
        "redirect_uri",
        "code_challenge",
        "code_challenge_method",
        "state",
        "scope",
        "resource",
    ):
        value = html.escape(params.get(name, ""), quote=True)
        hidden.append(f'<input type="hidden" name="{name}" value="{value}">')
    notice = f"<p>{html.escape(message)}</p>" if message else ""
    body = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Accesso Sinde</title>
</head>
<body>
  <h1>Accesso al server Sinde</h1>
  {notice}
  <form method="post" action="/oauth/authorize">
    {''.join(hidden)}
    <label>Chiave API<br><input type="password" name="api_key" autocomplete="current-password" required></label>
    <p><button type="submit">Consenti a ChatGPT</button></p>
  </form>
</body>
</html>"""
    return HTMLResponse(body)


def _validated_authorize_params(params: Dict[str, str]) -> Optional[tuple[str, Dict[str, Any]]]:
    if params.get("response_type") != "code":
        return None
    if params.get("code_challenge_method") != "S256" or not params.get("code_challenge"):
        return None
    resource = params.get("resource") or ""
    if resource.rstrip("/") not in ("", _origin(), _resource()):
        return None
    client_id = params.get("client_id") or ""
    redirect_uri = params.get("redirect_uri") or ""
    document = _load_cimd(client_id)
    if document is None or not _redirect_allowed(redirect_uri, document):
        return None
    return redirect_uri, document


def protected_resource_metadata() -> Dict[str, Any]:
    return {
        "resource": _resource(),
        "authorization_servers": [_origin()],
        "scopes_supported": ["sinde"],
        "bearer_methods_supported": ["header"],
    }


def authorization_server_metadata() -> Dict[str, Any]:
    origin = _origin()
    return {
        "issuer": origin,
        "authorization_endpoint": origin + "/oauth/authorize",
        "token_endpoint": origin + "/oauth/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "client_id_metadata_document_supported": True,
        "authorization_response_iss_parameter_supported": True,
        "scopes_supported": ["sinde"],
    }


def register_oauth_routes(mcp: Any) -> None:
    async def protected_resource(_request: Request) -> JSONResponse:
        return JSONResponse(protected_resource_metadata())

    async def authorization_server(_request: Request) -> JSONResponse:
        return JSONResponse(authorization_server_metadata())

    async def authorize(request: Request) -> Response:
        if not auth_enabled():
            return _oauth_error(404, "auth_not_configured")
        if request.method == "POST":
            form = await request.form()
            params = {key: str(form.get(key) or "") for key in (
                "response_type",
                "client_id",
                "redirect_uri",
                "code_challenge",
                "code_challenge_method",
                "state",
                "scope",
                "resource",
            )}
            supplied = str(form.get("api_key") or "")
        else:
            params = {key: request.query_params.get(key, "") for key in (
                "response_type",
                "client_id",
                "redirect_uri",
                "code_challenge",
                "code_challenge_method",
                "state",
                "scope",
                "resource",
            )}
            supplied = ""

        checked = _validated_authorize_params(params)
        if checked is None:
            return _oauth_error(400, "invalid_request")
        redirect_uri, _document = checked

        if request.method != "POST":
            return _form_page(params)

        host = _client_host(request)
        if len(supplied) > 512 or _too_many_failures(host) or not _secret_equal(supplied, api_key()):
            _record_failure(host)
            return _form_page(params, "Chiave non valida.")

        code = _store_code(params["code_challenge"], redirect_uri, params["client_id"])
        query = {"code": code, "iss": _origin()}
        if params.get("state"):
            query["state"] = params["state"]
        separator = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(redirect_uri + separator + urlencode(query), status_code=302)

    async def token(request: Request) -> JSONResponse:
        if not auth_enabled():
            return _oauth_error(404, "auth_not_configured")
        form = await request.form()
        if str(form.get("grant_type") or "") != "authorization_code":
            return _oauth_error(400, "unsupported_grant_type")
        code = str(form.get("code") or "")
        verifier = str(form.get("code_verifier") or "")
        redirect_uri = str(form.get("redirect_uri") or "")
        client_id = str(form.get("client_id") or "")
        stored = _take_code(code)
        if stored is None:
            return _oauth_error(400, "invalid_grant")
        if stored["redirect_uri"] != redirect_uri or stored["client_id"] != client_id:
            return _oauth_error(400, "invalid_grant")
        try:
            challenge = _pkce_s256(verifier) if 43 <= len(verifier) <= 128 else ""
        except UnicodeEncodeError:
            challenge = ""
        if challenge != stored["challenge"]:
            return _oauth_error(400, "invalid_grant")
        access_token = issue_access_token()
        return JSONResponse(
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "expires_in": _TOKEN_TTL_SECONDS,
                "scope": "sinde",
            },
            headers={"Cache-Control": "no-store"},
        )

    mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])(protected_resource)
    mcp.custom_route("/.well-known/oauth-protected-resource/mcp", methods=["GET"])(protected_resource)
    mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])(authorization_server)
    mcp.custom_route("/oauth/authorize", methods=["GET", "POST"])(authorize)
    mcp.custom_route("/oauth/token", methods=["POST"])(token)


def _is_mcp_path(path: str) -> bool:
    trimmed = path.rstrip("/") or "/"
    return trimmed == "/mcp" or path.startswith("/mcp/")


def _bearer_token(scope: Dict[str, Any]) -> str:
    for name, value in scope.get("headers") or []:
        if name.lower() == b"authorization":
            text = value.decode("latin-1")
            if text.lower().startswith("bearer "):
                return text[7:].strip()
    return ""


class McpBearerMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not auth_enabled():
            await self.app(scope, receive, send)
            return
        path = scope.get("path") or ""
        method = scope.get("method") or ""
        if method == "OPTIONS" or not _is_mcp_path(path):
            await self.app(scope, receive, send)
            return
        if access_token_valid(_bearer_token(scope)):
            await self.app(scope, receive, send)
            return
        body = b'{"error":"unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", _www_authenticate().encode("ascii")),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def install(app: Any) -> None:
    app.add_middleware(McpBearerMiddleware)
    if auth_enabled():
        print("[mcp-auth] MCP_API_KEY attiva: /mcp richiede il token OAuth")
    else:
        print("[mcp-auth] MCP_API_KEY assente: /mcp resta aperto")
