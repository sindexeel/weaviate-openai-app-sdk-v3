# serve.py
import os
import json
import re
import time
import uuid
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse
import mcp.types as types


from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

# --- Weaviate client imports (v4) ---
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

# OpenAI client per descrizioni immagini
from openai import OpenAI

_OPENAI_CLIENT = None
if os.environ.get("OPENAI_API_KEY"):
    _OPENAI_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
else:
    print("[query-caption] WARNING: OPENAI_API_KEY non impostata, niente descrizioni testuali per le query.")

# In-memory stato Vertex
_VERTEX_HEADERS: Dict[str, str] = {}
_VERTEX_REFRESH_THREAD_STARTED = False
_VERTEX_USER_PROJECT: Optional[str] = None

# In-memory storage per immagini caricate (temporaneo, scade dopo 1 ora)
_UPLOADED_IMAGES: Dict[str, Dict[str, Any]] = {}

# Ultimi risultati ricevuti dal widget Sinde (visibili ai tool MCP)
_LAST_WIDGET_RESULTS: Dict[str, Any] = {}

_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_PROMPT_PATH = _BASE_DIR / "prompts" / "instructions.md"
_DEFAULT_DESCRIPTION_PATH = _BASE_DIR / "prompts" / "description.txt"
_WIDGET_DIST_DIR = _BASE_DIR / "weaviate-image-app" / "dist"
_BASE_URL = (
    os.environ.get("PUBLIC_URL")
    or os.environ.get("BASE_URL")
    or "https://weaviate-openai-app-sdk-v3.onrender.com"
)
_BASE_HOST = urlparse(_BASE_URL).netloc


def _looks_like_pdf(file_bytes: bytes, filename: Optional[str] = None) -> bool:
    if filename and filename.lower().endswith(".pdf"):
        return True
    return file_bytes.startswith(b"%PDF-")


def _pdf_bytes_to_png_base64(file_bytes: bytes) -> Optional[str]:
    """
    Converte la prima pagina di un PDF in PNG base64.
    Restituisce None se la conversione fallisce.
    """
    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        print(f"[pdf] pypdfium2 non disponibile: {exc}")
        return None

    try:
        try:
            from PIL import Image  # noqa: F401
        except Exception as exc:
            print(f"[pdf] Pillow non disponibile per conversione PDF->PNG: {exc}")
            return None

        pdf = pdfium.PdfDocument(file_bytes)
        if len(pdf) == 0:
            print("[pdf] documento senza pagine")
            return None

        page = pdf[0]
        bitmap = page.render(scale=3).to_pil()
        from io import BytesIO

        buffer = BytesIO()
        bitmap.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[pdf] errore conversione PDF->PNG: {exc}")
        return None


def _build_vertex_header_map(token: str) -> Dict[str, str]:
    headers: Dict[str, str] = {
        "X-Goog-Vertex-Api-Key": token,
    }
        # user-project opzionale
    if _VERTEX_USER_PROJECT:
        headers["X-Goog-User-Project"] = _VERTEX_USER_PROJECT
    return headers


def _discover_gcp_project() -> Optional[str]:
    gac_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if gac_json:
        try:
            data = json.loads(gac_json)
            if isinstance(data, dict) and data.get("project_id"):
                return data["project_id"]
        except Exception:
            pass

    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_path and os.path.exists(gac_path):
        try:
            with open(gac_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("project_id"):
                return data["project_id"]
        except Exception:
            pass

    try:
        import google.auth

        creds, proj = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if proj:
            return proj
    except Exception:
        pass
    return None


def _get_weaviate_url() -> str:
    url = os.environ.get("WEAVIATE_CLUSTER_URL") or os.environ.get("WEAVIATE_URL")
    if not url:
        raise RuntimeError("Please set WEAVIATE_URL or WEAVIATE_CLUSTER_URL.")
    return url


def _get_weaviate_api_key() -> str:
    api_key = os.environ.get("WEAVIATE_API_KEY")
    if not api_key:
        raise RuntimeError("Please set WEAVIATE_API_KEY.")
    return api_key


def _get_default_collection() -> str:
    """
    Restituisce il nome della collection di default.
    Se WEAVIATE_DEFAULT_COLLECTION è impostata, usa quella; altrimenti 'Sinde3'.
    """
    return os.environ.get("WEAVIATE_DEFAULT_COLLECTION", "Sinde2")


def _get_default_alpha() -> float:
    """
    Restituisce l'alpha di default per hybrid_search.
    Se HYBRID_DEFAULT_ALPHA è impostata (env), usa quella; altrimenti 0.7.
    """
    val = os.environ.get("HYBRID_DEFAULT_ALPHA")
    if not val:
        return 0.7
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.7


def _resolve_service_account_path() -> Optional[str]:
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_path and os.path.exists(gac_path):
        _load_vertex_user_project(gac_path)
        return gac_path

    candidates = [
        os.environ.get("VERTEX_SA_PATH"),
        "/etc/secrets/weaviate-sa.json",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = candidate
            _load_vertex_user_project(candidate)
            return candidate
    return None


def _load_vertex_user_project(path: str) -> None:
    global _VERTEX_USER_PROJECT
    if _VERTEX_USER_PROJECT:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _VERTEX_USER_PROJECT = data.get("project_id")
        if not _VERTEX_USER_PROJECT and data.get("quota_project_id"):
            _VERTEX_USER_PROJECT = data["quota_project_id"]
        if _VERTEX_USER_PROJECT:
            try:
                print(
                    f"[vertex-oauth] detected service account project: {_VERTEX_USER_PROJECT}"
                )
            except (ValueError, OSError):
                pass
        else:
            try:
                print(
                    "[vertex-oauth] warning: project_id not found in service account JSON"
                )
            except (ValueError, OSError):
                pass
    except Exception as exc:
        try:
            print(f"[vertex-oauth] unable to read project id from SA: {exc}")
        except (ValueError, OSError):
            pass


def _sync_refresh_vertex_token() -> bool:
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
    except Exception as exc:
        print(f"[vertex-oauth] sync refresh unavailable: {exc}")
        return False

    cred_path = _resolve_service_account_path()
    if not cred_path or not os.path.exists(cred_path):
        return False
    try:
        creds = service_account.Credentials.from_service_account_file(
            cred_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        creds.refresh(Request())
    except Exception as exc:
        print(f"[vertex-oauth] sync refresh error: {exc}")
        return False

    token = creds.token
    if not token:
        return False
    global _VERTEX_HEADERS
    _VERTEX_HEADERS = _build_vertex_header_map(token)
    print(f"[vertex-oauth] sync token refresh (prefix: {token[:10]}...)")
    # Aggiorna anche le variabili d'ambiente per Weaviate vectorizer
    os.environ["GOOGLE_APIKEY"] = token
    os.environ["PALM_APIKEY"] = token
    return True


def _connect():
    url = _get_weaviate_url()
    key = _get_weaviate_api_key()
    _resolve_service_account_path()

    headers: Dict[str, str] = {}
    
    # OpenAI (se ti serve per text2vec-openai / altre cose)
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    if openai_key:
        headers["X-OpenAI-Api-Key"] = openai_key

    # 👇 QUI la parte importante: esattamente come fai in Colab
    # Prendi il token OAuth Vertex da _VERTEX_HEADERS (aggiornato dal refresh)
    vertex_token = (
        os.environ.get("VERTEX_APIKEY")
        or os.environ.get("VERTEX_BEARER_TOKEN")
    )
    
    # Se non c'è una chiave statica/bearer, usa OAuth
    if not vertex_token:
        # Assicurati che _VERTEX_HEADERS contenga il token
        if not ("_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS and _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key")):
            # Se _VERTEX_HEADERS è vuoto o non contiene il token, facciamo un refresh
            _sync_refresh_vertex_token()
        # Prendi il token da _VERTEX_HEADERS
        if "_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS:
            vertex_token = _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key")
    
    if vertex_token:
        # Esattamente come nel Colab: metti il token nell'header
        headers["X-Goog-Vertex-Api-Key"] = vertex_token
        # Aggiorna anche le variabili d'ambiente per Weaviate vectorizer (fallback)
        os.environ["GOOGLE_APIKEY"] = vertex_token
        os.environ["PALM_APIKEY"] = vertex_token
        print(f"[vertex-oauth] using Vertex token (prefix: {vertex_token[:10]}...)")
    else:
        print("[vertex-oauth] WARNING: no Vertex token available for connection")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(key),
        headers=headers or None,
    )

    # Imposta anche i metadata gRPC (necessari per Weaviate)
    grpc_meta: Dict[str, str] = {}
    if vertex_token:
        grpc_meta["x-goog-vertex-api-key"] = vertex_token
        if _VERTEX_USER_PROJECT:
            grpc_meta["x-goog-user-project"] = _VERTEX_USER_PROJECT
    if openai_key:
        grpc_meta["x-openai-api-key"] = openai_key

    try:
        conn = getattr(client, "_connection", None)
        if conn is not None:
            meta_list = list(grpc_meta.items())
            try:
                setattr(conn, "grpc_metadata", meta_list)
            except Exception:
                pass
            try:
                setattr(conn, "_grpc_metadata", meta_list)
            except Exception:
                pass
            if hasattr(conn, "set_grpc_metadata"):
                try:
                    conn.set_grpc_metadata(meta_list)
                except Exception:
                    pass
            debug_meta = getattr(conn, "grpc_metadata", None)
            print(f"[vertex-oauth] grpc metadata now: {debug_meta}")
    except Exception as e:
        print("[weaviate] warning: cannot set gRPC metadata headers:", e)

    return client


def _update_client_grpc_metadata(client):
    """Aggiorna i metadata gRPC del client con le credenziali Vertex più recenti."""
    try:
        grpc_meta: Dict[str, str] = {}
        
        # Assicuriamoci che _VERTEX_HEADERS contenga il token più recente
        if not ("_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS and _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key")):
            # Se _VERTEX_HEADERS è vuoto o non contiene il token, facciamo un refresh
            _sync_refresh_vertex_token()
        
        # Aggiungiamo i metadata Vertex se disponibili
        if "_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS:
            vertex_token = _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key") or _VERTEX_HEADERS.get("x-goog-vertex-api-key")
            if vertex_token:
                grpc_meta["x-goog-vertex-api-key"] = vertex_token
                # Aggiorna anche le variabili d'ambiente per Weaviate vectorizer
                os.environ["GOOGLE_APIKEY"] = vertex_token
                os.environ["PALM_APIKEY"] = vertex_token
            if _VERTEX_USER_PROJECT:
                grpc_meta["x-goog-user-project"] = _VERTEX_USER_PROJECT
            auth = _VERTEX_HEADERS.get("Authorization") or _VERTEX_HEADERS.get("authorization")
            if auth:
                grpc_meta["authorization"] = auth
        
        # Aggiungiamo anche OpenAI key se disponibile
        openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
        if openai_key:
            grpc_meta["x-openai-api-key"] = openai_key
        
        if not grpc_meta:
            return
        
        # Aggiorna i metadata gRPC del client
        conn = getattr(client, "_connection", None)
        if conn is not None:
            meta_list = list(grpc_meta.items())
            try:
                setattr(conn, "grpc_metadata", meta_list)
            except Exception:
                pass
            try:
                setattr(conn, "_grpc_metadata", meta_list)
            except Exception:
                pass
            if hasattr(conn, "set_grpc_metadata"):
                try:
                    conn.set_grpc_metadata(meta_list)
                except Exception:
                    pass
            
            # Prova anche ad aggiornare gli header REST se possibile
            try:
                vertex_token = None
                if "_VERTEX_HEADERS" in globals() and _VERTEX_HEADERS:
                    vertex_token = _VERTEX_HEADERS.get("X-Goog-Vertex-Api-Key") or _VERTEX_HEADERS.get("x-goog-vertex-api-key")
                
                if vertex_token:
                    # Prova vari attributi possibili per gli header REST
                    for attr_name in ["_headers", "headers", "_rest_headers", "rest_headers"]:
                        if hasattr(conn, attr_name):
                            headers_attr = getattr(conn, attr_name)
                            if headers_attr is not None and isinstance(headers_attr, dict):
                                headers_attr["X-Goog-Vertex-Api-Key"] = vertex_token
                                if _VERTEX_USER_PROJECT:
                                    headers_attr["X-Goog-User-Project"] = _VERTEX_USER_PROJECT
                                break
            except Exception:
                pass
    except Exception as e:
        print(f"[vertex-oauth] warning: cannot update gRPC metadata: {e}")


def _load_text_source(env_keys, file_path):
    if isinstance(env_keys, str):
        env_keys = [env_keys]
    path = Path(file_path) if file_path else None
    if path and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as exc:
            print(f"[mcp] warning: cannot read instructions file '{path}': {exc}")
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            return val.strip()
    return None


_MCP_SERVER_NAME = os.environ.get("MCP_SERVER_NAME", "weaviate-mcp-http")
_MCP_INSTRUCTIONS_FILE = os.environ.get("MCP_PROMPT_FILE") or os.environ.get(
    "MCP_INSTRUCTIONS_FILE"
)
if not _MCP_INSTRUCTIONS_FILE and _DEFAULT_PROMPT_PATH.exists():
    _MCP_INSTRUCTIONS_FILE = str(_DEFAULT_PROMPT_PATH)
_MCP_DESCRIPTION_FILE = os.environ.get("MCP_DESCRIPTION_FILE")
if not _MCP_DESCRIPTION_FILE and _DEFAULT_DESCRIPTION_PATH.exists():
    _MCP_DESCRIPTION_FILE = str(_DEFAULT_DESCRIPTION_PATH)

_MCP_INSTRUCTIONS = _load_text_source(
    ["MCP_PROMPT", "MCP_INSTRUCTIONS"], _MCP_INSTRUCTIONS_FILE
)
_MCP_DESCRIPTION = _load_text_source("MCP_DESCRIPTION", _MCP_DESCRIPTION_FILE)

# Porta e host per FastMCP / uvicorn (per Render)
SERVER_PORT = int(os.environ.get("PORT", "10000"))
os.environ.setdefault("FASTMCP_PORT", str(SERVER_PORT))
os.environ.setdefault("FASTMCP_HOST", "0.0.0.0")

# Configura la transport security direttamente sul costruttore FastMCP (approccio ufficiale mcp 1.23+).
# Necessario su Render: host=0.0.0.0 attiva la protezione DNS rebinding con allowed_hosts=[],
# bloccando tutte le richieste con 421. Passiamo l'host pubblico esplicitamente.
_transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "localhost",
        "localhost:*",
        "127.0.0.1",
        "127.0.0.1:*",
        _BASE_HOST,
        f"{_BASE_HOST}:*",
    ],
)
print(f"[mcp] TransportSecuritySettings.allowed_hosts = {_transport_security.allowed_hosts}")

mcp = FastMCP(_MCP_SERVER_NAME, stateless_http=True, transport_security=_transport_security)


def _apply_mcp_metadata():
    try:
        if hasattr(mcp, "set_server_info"):
            server_info: Dict[str, Any] = {}
            if _MCP_DESCRIPTION:
                server_info["description"] = _MCP_DESCRIPTION
            if _MCP_INSTRUCTIONS:
                server_info["instructions"] = _MCP_INSTRUCTIONS
            if server_info:
                mcp.set_server_info(**server_info)
    except Exception:
        pass


_apply_mcp_metadata()


def _load_widget_html() -> str:
    widget_html_path = _WIDGET_DIST_DIR / "index.html"

    if not widget_html_path.exists():
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Search Widget</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="{_BASE_URL}/assets/index.js"></script>
</body>
</html>"""

    try:
        with open(widget_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Vite genera path come /assets/index-xxx.js (con base: '/assets/')
        # Dobbiamo sostituire /assets/ con {_BASE_URL}/assets/ senza creare doppio assets
        import re
        
        # Prima rimuovi eventuali doppi assets (correzione per path già modificati)
        base_url_escaped = _BASE_URL.replace('/', r'\/')
        html_content = re.sub(
            rf'{base_url_escaped}/assets/assets/',
            rf'{_BASE_URL}/assets/',
            html_content
        )
        
        # Sostituisce src="/assets/..." con src="{_BASE_URL}/assets/..."
        html_content = re.sub(
            r'src="/assets/([^"]+)"',
            rf'src="{_BASE_URL}/assets/\1"',
            html_content
        )
        html_content = re.sub(
            r'href="/assets/([^"]+)"',
            rf'href="{_BASE_URL}/assets/\1"',
            html_content
        )
        # Gestisce anche path relativi assets/... (senza slash iniziale)
        html_content = re.sub(
            r'src="assets/([^"]+)"',
            rf'src="{_BASE_URL}/assets/\1"',
            html_content
        )
        html_content = re.sub(
            r'href="assets/([^"]+)"',
            rf'href="{_BASE_URL}/assets/\1"',
            html_content
        )

        return html_content
    except Exception as e:
        print(f"[widget] Error loading widget HTML: {e}")
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Search Widget</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="{_BASE_URL}/assets/index.js"></script>
</body>
</html>"""


@dataclass(frozen=True)
class SindeWidget:
    identifier: str
    title: str
    template_uri: str
    invoking: str
    invoked: str
    html: str
    response_text: str


@lru_cache(maxsize=None)
def _load_widget_html_cached() -> str:
    return _load_widget_html()


widget_uri = "ui://widget/image-search.html"


SINDE_WIDGET = SindeWidget(
    identifier="open_image_search_widget",          # nome del tool
    title="Ricerca progetti Sinde",                 # come vuoi vederlo in ChatGPT
    template_uri=widget_uri,
    invoking="Apro il widget di ricerca progetti Sinde...",
    invoked="Widget di ricerca progetti Sinde pronto.",
    html=_load_widget_html_cached(),                # HTML già buildato
    response_text="Ho aperto il widget di ricerca immagini Sinde.",
)


MIME_TYPE = "text/html+skybridge"


def _tool_meta(widget: SindeWidget) -> Dict[str, Any]:
    return {
        "openai/outputTemplate": widget.template_uri,
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
        "openai/widgetAccessible": True,
        "openai/resultCanProduceWidget": True,
    }


def _resource_description(widget: SindeWidget) -> str:
    return f"{widget.title} widget markup"


@mcp.resource(
    uri=widget_uri,
    name="image-search-widget",
    description="Widget per la ricerca di immagini in Weaviate",
)
def image_search_widget_resource():
    widget_html = _load_widget_html()
    return {
        "contents": [
            {
                "uri": widget_uri,
                "mimeType": "text/html+skybridge",
                "text": widget_html,
                "_meta": {
                    "openai/widgetPrefersBorder": True,
                    "openai/widgetDomain": "https://chatgpt.com",
                    "openai/widgetCSP": {
                        "connect_domains": [_BASE_URL],
                        "resource_domains": ["https://*.oaistatic.com"],
                    },
                },
            }
        ],
    }


# @mcp.tool()
# def open_image_search_widget() -> Dict[str, Any]:
#     """
#     Apre il widget interattivo per la ricerca di immagini.
#     """
#     return {
#         "structuredContent": {
#             "widgetReady": True,
#             "message": "Widget di ricerca immagini pronto all'uso.",
#         },
#         "content": [
#             {
#                 "type": "text",
#                 "text": (
#                     "Ho aperto il widget di ricerca immagini. "
#                     "Puoi caricare un'immagine e cercare immagini simili nella collection Sinde."
#                 ),
#             }
#         ],
#         "_meta": {
#             "baseUrl": _BASE_URL,
#         },
#     }


# def _add_tool_metadata():
#     try:
#         if hasattr(mcp, "_tools"):
#             tools = mcp._tools
#         elif hasattr(mcp, "tools"):
#             tools = mcp.tools
#         else:
#             app = getattr(mcp, "app", None) or getattr(mcp, "_app", None)
#             if app and hasattr(app, "state") and hasattr(app.state, "tools"):
#                 tools = app.state.tools
#             else:
#                 return
#
#         tool_name = "open_image_search_widget"
#         if isinstance(tools, dict) and tool_name in tools:
#             tool_def = tools[tool_name]
#             meta = getattr(tool_def, "_meta", None)
#             if not isinstance(meta, dict):
#                 meta = {}
#             meta.update(
#                 {
#                     "openai/outputTemplate": widget_uri,
#                     "openai/toolInvocation/invoking": (
#                         "Aprendo il widget di ricerca immagini..."
#                     ),
#                     "openai/toolInvocation/invoked": (
#                         "Widget di ricerca immagini pronto."
#                     ),
#                 }
#             )
#             tool_def._meta = meta
#         elif isinstance(tools, list):
#             for tool_def in tools:
#                 if getattr(tool_def, "name", None) == tool_name:
#                     meta = getattr(tool_def, "_meta", None)
#                     if not isinstance(meta, dict):
#                         meta = {}
#                     meta.update(
#                         {
#                             "openai/outputTemplate": widget_uri,
#                             "openai/toolInvocation/invoking": (
#                                 "Aprendo il widget di ricerca immagini..."
#                             ),
#                             "openai/toolInvocation/invoked": (
#                                 "Widget di ricerca immagini pronto."
#                             ),
#                         }
#                     )
#                     tool_def._meta = meta
#                     break
#     except Exception as e:
#         print(f"[widget] Warning: Could not add metadata to tool: {e}")
#
#
# _add_tool_metadata()


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    return JSONResponse({"status": "ok", "service": "weaviate-mcp-http"})


@mcp.custom_route("/test_cases.json", methods=["GET"])
async def serve_test_cases(request):
    from starlette.responses import Response
    file_path = _WIDGET_DIST_DIR / "test_cases.json"
    if not file_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return Response(
        file_path.read_text(encoding="utf-8"),
        media_type="application/json",
        headers={"Cache-Control": "no-cache"},
    )


@mcp.custom_route("/assets/{file_path:path}", methods=["GET"])
async def serve_assets(request):
    from starlette.responses import FileResponse

    file_path = request.path_params.get("file_path", "")
    
    # Rimuovi eventuale prefisso "assets/" duplicato
    if file_path.startswith("assets/"):
        file_path = file_path[7:]  # Rimuovi "assets/"

    full_path = _WIDGET_DIST_DIR / "assets" / file_path
    if not full_path.exists():
        full_path = _WIDGET_DIST_DIR / file_path

    try:
        resolved_path = full_path.resolve()
        dist_resolved = _WIDGET_DIST_DIR.resolve()
        resolved_path.relative_to(dist_resolved)
    except (ValueError, OSError):
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    if not full_path.exists() or not full_path.is_file():
        return JSONResponse({"error": "Not found"}, status_code=404)

    content_type_map = {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".html": "text/html",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".ttf": "font/ttf",
        ".eot": "application/vnd.ms-fontobject",
    }

    ext = full_path.suffix.lower()
    content_type = content_type_map.get(ext, "application/octet-stream")

    return FileResponse(
        full_path,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=31536000",
            "Access-Control-Allow-Origin": "*",
        },
    )


@mcp.custom_route("/upload-image", methods=["POST"])
async def upload_image_endpoint(request):
    """
    Endpoint HTTP per upload diretto di immagini.
    """
    try:
        content_type = request.headers.get("content-type", "")
        image_b64 = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            if "image" not in form:
                return JSONResponse(
                    {"error": "Missing 'image' field in form data"}, status_code=400
                )

            file = form["image"]
            if hasattr(file, "read"):
                try:
                    file_bytes = await file.read()
                    filename = getattr(file, "filename", None)
                    if _looks_like_pdf(file_bytes, filename):
                        image_b64 = _pdf_bytes_to_png_base64(file_bytes)
                        if not image_b64:
                            return JSONResponse(
                                {
                                    "error": (
                                        "Impossibile processare il PDF. "
                                        "Installa pypdfium2 e pillow e verifica che il file non sia corrotto."
                                    )
                                },
                                status_code=400,
                            )
                    else:
                        image_b64 = base64.b64encode(file_bytes).decode("utf-8")
                finally:
                    try:
                        await file.close()
                    except Exception:
                        pass
            else:
                return JSONResponse(
                    {"error": "Invalid file upload"}, status_code=400
                )
        else:
            try:
                data = await request.json()
                image_b64 = data.get("image_b64")
                if not image_b64:
                    return JSONResponse(
                        {"error": "Missing 'image_b64' in JSON body"}, status_code=400
                    )
            except Exception:
                return JSONResponse(
                    {
                        "error": (
                            "Invalid request format. Use multipart/form-data with "
                            "'image' field or JSON with 'image_b64'"
                        )
                    },
                    status_code=400,
                )

        if not image_b64:
            return JSONResponse(
                {"error": "No image data provided"}, status_code=400
            )

        cleaned_b64 = _clean_base64(image_b64)
        if not cleaned_b64:
            return JSONResponse(
                {"error": "Invalid base64 image string"}, status_code=400
            )

        image_id = str(uuid.uuid4())

        _UPLOADED_IMAGES[image_id] = {
            "image_b64": cleaned_b64,
            "expires_at": time.time() + 3600,
        }

        current_time = time.time()
        expired_ids = [
            img_id
            for img_id, data in _UPLOADED_IMAGES.items()
            if data["expires_at"] < current_time
        ]
        for img_id in expired_ids:
            _UPLOADED_IMAGES.pop(img_id, None)

        return JSONResponse({"image_id": image_id, "expires_in": 3600})
    except Exception as e:
        print(f"[upload-image] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/image-search", methods=["POST"])
async def image_search_http(request):
    """
    Endpoint HTTP per effettuare la ricerca immagini usando hybrid_search.
    Si aspetta un JSON tipo:
      {
        "collection": "Sinde3",
        "image_id": "uuid from /upload-image",
        "image_url": "... (opzionale)",
        "caption": "... (opzionale, non più usato)",
        "limit": 20
      }
    
    Usa hybrid_search che genera il vettore esternamente con Vertex AI + GPT,
    invece di image_search_vertex che richiede un vectorizer interno di Weaviate.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    # Usa la collection passata o il default da WEAVIATE_DEFAULT_COLLECTION (fallback 'Sinde3')
    collection = data.get("collection") or _get_default_collection()
    image_id = data.get("image_id")
    image_url = data.get("image_url")
    limit = data.get("limit") or 50

    if not image_id and not image_url:
        return JSONResponse(
            {"error": "Either image_id or image_url must be provided"},
            status_code=400,
        )

    try:
        # Usa hybrid_search invece di image_search_vertex
        # hybrid_search genera il vettore esternamente con Vertex AI + GPT
        result = hybrid_search(
            collection=collection,
            query="",  # niente testo utente, è una pura ricerca per immagine
            limit=limit,
            # Usa il default alpha=0.2 di hybrid_search (20% vettoriale, 80% BM25)
            query_properties=["caption", "source_pdf"],
            image_id=image_id,
            image_url=image_url,
        )
        return JSONResponse(result)
    except Exception as e:
        print(f"[image-search-http] error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/widget-push-results", methods=["POST"])
async def widget_push_results(request):
    """
    Endpoint HTTP chiamato SOLO dal widget per salvare gli ultimi risultati
    di ricerca, in modo che un tool MCP possa poi restituirli a ChatGPT.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    summary = data.get("results_summary")
    raw = data.get("raw_results")

    if not summary:
        return JSONResponse(
            {"error": "results_summary is required"},
            status_code=400,
        )

    # Salva in memoria globale
    global _LAST_WIDGET_RESULTS
    _LAST_WIDGET_RESULTS["summary"] = summary
    _LAST_WIDGET_RESULTS["raw_results"] = raw

    return JSONResponse({"ok": True})


@mcp.tool()
def get_instructions() -> Dict[str, Any]:
    return {
        "instructions": _MCP_INSTRUCTIONS,
        "description": _MCP_DESCRIPTION,
        "server_name": _MCP_SERVER_NAME,
        "prompt_file": _MCP_INSTRUCTIONS_FILE,
        "description_file": _MCP_DESCRIPTION_FILE,
    }


@mcp.tool()
def reload_instructions() -> Dict[str, Any]:
    global _MCP_INSTRUCTIONS, _MCP_DESCRIPTION, _MCP_INSTRUCTIONS_FILE, _MCP_DESCRIPTION_FILE
    _MCP_INSTRUCTIONS_FILE = os.environ.get("MCP_PROMPT_FILE") or os.environ.get(
        "MCP_INSTRUCTIONS_FILE"
    )
    if not _MCP_INSTRUCTIONS_FILE and _DEFAULT_PROMPT_PATH.exists():
        _MCP_INSTRUCTIONS_FILE = str(_DEFAULT_PROMPT_PATH)
    _MCP_DESCRIPTION_FILE = os.environ.get("MCP_DESCRIPTION_FILE")
    if not _MCP_DESCRIPTION_FILE and _DEFAULT_DESCRIPTION_PATH.exists():
        _MCP_DESCRIPTION_FILE = str(_DEFAULT_DESCRIPTION_PATH)
    _MCP_INSTRUCTIONS = _load_text_source(
        ["MCP_PROMPT", "MCP_INSTRUCTIONS"], _MCP_INSTRUCTIONS_FILE
    )
    _MCP_DESCRIPTION = _load_text_source("MCP_DESCRIPTION", _MCP_DESCRIPTION_FILE)
    _apply_mcp_metadata()
    return get_instructions()


@mcp.tool()
def get_config() -> Dict[str, Any]:
    return {
        "weaviate_url": os.environ.get("WEAVIATE_CLUSTER_URL")
        or os.environ.get("WEAVIATE_URL"),
        "weaviate_api_key_set": bool(os.environ.get("WEAVIATE_API_KEY")),
        "default_collection": _get_default_collection(),
        "default_alpha": _get_default_alpha(),
        "prompt_file": _MCP_INSTRUCTIONS_FILE,
        "openai_api_key_set": bool(
            os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
        ),
        "cohere_api_key_set": bool(os.environ.get("COHERE_API_KEY")),
    }


@mcp.tool()
def debug_widget() -> Dict[str, Any]:
    widget_html_path = _WIDGET_DIST_DIR / "index.html"
    widget_exists = widget_html_path.exists()
    assets_dir = _WIDGET_DIST_DIR / "assets"
    assets_exist = assets_dir.exists() if assets_dir else False

    return {
        "widget_html_exists": widget_exists,
        "widget_html_path": str(widget_html_path),
        "assets_dir_exists": assets_exist,
        "base_url": _BASE_URL,
        "widget_template_uri": widget_uri,
        "widget_identifier": "image-search-widget",
    }


@mcp.tool()
def sinde_widget_push_results(
    results_summary: Optional[str] = None,
    raw_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tool MCP che restituisce gli ultimi risultati salvati dal widget.

    - Uso normale (ChatGPT che lo chiama): ignora gli argomenti e
      restituisce quello che il widget ha mandato a /widget-push-results.
    - Se per qualche motivo non c'è nulla in memoria, usa gli argomenti
      passati (fallback, utile in test).
    """
    # Se il widget ha già pushato qualcosa via /widget-push-results
    if _LAST_WIDGET_RESULTS:
        return {
            "summary": _LAST_WIDGET_RESULTS.get("summary"),
            "raw_results": _LAST_WIDGET_RESULTS.get("raw_results"),
        }

    # Fallback: usa gli argomenti (per compatibilità)
    return {
        "summary": results_summary,
        "raw_results": raw_results,
    }


@mcp.tool()
def get_last_sinde_results() -> Dict[str, Any]:
    """
    Restituisce gli ultimi risultati che il widget Sinde ha salvato tramite /widget-push-results.
    
    - Nessun argomento richiesto.
    - Se non ci sono risultati, restituisce summary/raw_results = None.
    """
    if not _LAST_WIDGET_RESULTS:
        return {
            "summary": None,
            "raw_results": None,
        }

    return {
        "summary": _LAST_WIDGET_RESULTS.get("summary"),
        "raw_results": _LAST_WIDGET_RESULTS.get("raw_results"),
    }


@mcp.tool()
def check_connection() -> Dict[str, Any]:
    client = _connect()
    try:
        ready = client.is_ready()
        return {"ready": bool(ready)}
    finally:
        client.close()


@mcp.tool()
def upload_image(
    image_url: Optional[str] = None, image_path: Optional[str] = None
) -> Dict[str, Any]:
    global _UPLOADED_IMAGES

    cleaned_b64 = None

    if image_path:
        print(f"[upload_image] Loading image from path: {image_path}")
        try:
            if not os.path.exists(image_path):
                return {"error": f"File not found: {image_path}"}
            with open(image_path, "rb") as f:
                file_bytes = f.read()

            if _looks_like_pdf(file_bytes, image_path):
                image_b64_raw = _pdf_bytes_to_png_base64(file_bytes)
                if not image_b64_raw:
                    return {
                        "error": (
                            "Failed to convert PDF to image. "
                            "Install pypdfium2 and verify the PDF file."
                        )
                    }
            else:
                image_b64_raw = base64.b64encode(file_bytes).decode("utf-8")

            cleaned_b64 = _clean_base64(image_b64_raw) if image_b64_raw else None
        except Exception as e:
            return {
                "error": f"Failed to load image from path {image_path}: {str(e)}"
            }
        if not cleaned_b64:
            return {"error": f"Invalid image file: {image_path}"}
    elif image_url:
        print(f"[upload_image] Loading image from URL: {image_url}")
        cleaned_b64 = _load_image_from_url(image_url)
        if not cleaned_b64:
            return {"error": f"Failed to load image from URL: {image_url}"}
    else:
        return {"error": "Either image_url or image_path must be provided"}

    image_id = str(uuid.uuid4())

    _UPLOADED_IMAGES[image_id] = {
        "image_b64": cleaned_b64,
        "expires_at": time.time() + 3600,
    }

    current_time = time.time()
    expired_ids = [
        img_id
        for img_id, data in _UPLOADED_IMAGES.items()
        if data["expires_at"] < current_time
    ]
    for img_id in expired_ids:
        _UPLOADED_IMAGES.pop(img_id, None)

    return {"image_id": image_id, "expires_in": 3600}


@mcp.tool()
def list_collections() -> List[str]:
    client = _connect()
    try:
        colls = client.collections.list_all()
        if isinstance(colls, dict):
            names = list(colls.keys())
        else:
            try:
                names = [getattr(c, "name", str(c)) for c in colls]
            except Exception:
                names = list(colls)
        return sorted(set(names))
    finally:
        client.close()


@mcp.tool()
def get_schema(collection: str) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        try:
            cfg = coll.config.get()
        except Exception:
            try:
                cfg = coll.config.get_class()
            except Exception:
                cfg = {"info": "config API not available in this client version"}
        return {"collection": collection, "config": cfg}
    finally:
        client.close()


@mcp.tool()
def keyword_search(collection: str, query: str, limit: int = 10) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        resp = coll.query.bm25(
            query=query,
            return_metadata=MetadataQuery(score=True),
            limit=limit,
        )
        out = []
        for o in getattr(resp, "objects", []) or []:
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "bm25_score": getattr(getattr(o, "metadata", None), "score", None),
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


@mcp.tool()
def semantic_search(collection: str, query: str, limit: int = 10) -> Dict[str, Any]:
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        resp = coll.query.near_text(
            query=query,
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        out = []
        for o in getattr(resp, "objects", []) or []:
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "distance": getattr(getattr(o, "metadata", None), "distance", None),
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


@mcp.tool()
def hybrid_search(
    collection: str,
    query: str,
    limit: int = 20,
    alpha: Optional[float] = None,
    query_properties: Optional[Any] = None,
    image_id: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    # Se alpha non è specificato, usa il default da env (HYBRID_DEFAULT_ALPHA) o 0.2
    if alpha is None:
        alpha = _get_default_alpha()

    # Usa la collection di default configurata, mantenendo lo stesso comportamento di forzatura
    default_collection = _get_default_collection()
    if not collection:
        collection = default_collection
    elif collection != default_collection:
        print(
            f"[hybrid_search] warning: collection '{collection}' requested, "
            f"but using '{default_collection}' as per instructions"
        )
        collection = default_collection

    if query_properties and isinstance(query_properties, str):
        try:
            query_properties = json.loads(query_properties)
        except (json.JSONDecodeError, TypeError):
            pass

    image_b64 = None

    if image_id:
        if image_id in _UPLOADED_IMAGES:
            img_data = _UPLOADED_IMAGES[image_id]
            if img_data["expires_at"] > time.time():
                image_b64 = img_data["image_b64"]
            else:
                _UPLOADED_IMAGES.pop(image_id, None)
                return {
                    "error": (
                        f"Image ID {image_id} has expired. Please upload the image again."
                    )
                }
        else:
            return {
                "error": (
                    f"Image ID {image_id} not found. Please upload the image first using upload_image."
                )
            }

    if image_url and not image_b64:
        image_b64 = _load_image_from_url(image_url)
        if not image_b64:
            return {"error": f"Failed to load image from URL: {image_url}"}
        image_b64 = _clean_base64(image_b64)
        if not image_b64:
            return {"error": f"Invalid image format from URL: {image_url}"}

    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}

        # Aggiorna i metadata gRPC prima della query per assicurarci che siano aggiornati
        _update_client_grpc_metadata(client)

        if image_b64:
            # Fusion search: DINOv2 + Caption (se abilitato)
            if _FUSION_ENABLED:
                try:
                    result = _do_fusion_search(client, image_b64, limit)
                    return result
                except Exception as exc:
                    print(f"[fusion] failed, falling back to caption-only: {exc}")
                    import traceback
                    traceback.print_exc()

            # Fallback: caption-only hybrid search
            query_caption = describe_image_for_query(image_b64) or ""

            print(f"[DEBUG] query_caption (len={len(query_caption)}): {query_caption[:120]}...")

            final_query = (
                query_caption
                or query
                or "technical mechanical drawing component geometry"
            )

            hybrid_params: Dict[str, Any] = {
                "query": final_query,
                "alpha": alpha,
                "limit": limit,
                "return_properties": ["source_pdf", "caption", "image_b64"],
                "return_metadata": MetadataQuery(score=True, distance=True),
            }
            hybrid_params["query_properties"] = ["caption"]

            print(f"[DEBUG] hybrid_params: query={repr(hybrid_params['query'][:80])}, alpha={hybrid_params['alpha']}, limit={hybrid_params['limit']}")

            try:
                resp = coll.query.hybrid(**hybrid_params)
            except Exception as exc:
                err_text = str(exc)
                if (
                    "No embedding input is provided" in err_text
                    or "remote client vectorize" in err_text
                ):
                    # Fallback robusto: se il vectorizer remoto rifiuta la query,
                    # usiamo BM25 per non bloccare la ricerca lato widget.
                    bm25_query = final_query.strip() or "mechanical drawing"
                    print(
                        "[hybrid_search] hybrid vectorize failed; fallback to bm25 "
                        f"with query={repr(bm25_query)}"
                    )
                    resp = coll.query.bm25(
                        query=bm25_query,
                        query_properties=["caption"],
                        limit=limit,
                        return_properties=[
                            "source_pdf",
                            "caption",
                            "image_b64",
                        ],
                        return_metadata=MetadataQuery(score=True),
                    )
                else:
                    raise
        else:
            hybrid_params = {
                "query": query,
                "alpha": alpha,
                "limit": limit,
                "return_properties": ["source_pdf", "caption", "image_b64"],
                "return_metadata": MetadataQuery(score=True, distance=True),
            }
            if query_properties:
                hybrid_params["query_properties"] = query_properties
            resp = coll.query.hybrid(**hybrid_params)

        # Log dei risultati nel formato Colab
        print("[DEBUG] Risultati hybrid search:")
        for o in getattr(resp, "objects", []) or []:
            name = getattr(o, "properties", {}).get("source_pdf", "N/A")
            md = getattr(o, "metadata", None)
            score = getattr(md, "score", None)
            if score is not None:
                print(f"{name}  score={score:.4f}")
            else:
                print(f"{name}  score=N/A")

        out = []
        for o in getattr(resp, "objects", []) or []:
            md = getattr(o, "metadata", None)
            score = getattr(md, "score", None)
            distance = getattr(md, "distance", None)
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "bm25_score": score,
                    "distance": distance,
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


try:
    from google.cloud import aiplatform

    _VERTEX_AVAILABLE = True
except Exception:
    _VERTEX_AVAILABLE = False


def _ensure_gcp_adc():
    gac_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_json and not gac_path:
        tmp_path = "/app/gcp_credentials.json"
        with open(tmp_path, "w", encoding="utf-8") as f2:
            f2.write(gac_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path
    _resolve_service_account_path()
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        _load_vertex_user_project(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])


def _load_image_from_url(image_url: str) -> Optional[str]:
    try:
        import requests
        import base64

        response = requests.get(image_url, timeout=30, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            print(
                f"[image] warning: URL {image_url} does not return an image (content-type: {content_type})"
            )

        content = response.content
        if len(content) > 10 * 1024 * 1024:
            print(
                f"[image] warning: image from {image_url} is too large ({len(content)} bytes)"
            )
            return None

        if len(content) < 100:
            print(
                f"[image] warning: image from {image_url} is too small ({len(content)} bytes)"
            )
            return None

        valid_formats = {
            b"\xff\xd8\xff": "JPEG",
            b"\x89PNG\r\n\x1a\n": "PNG",
            b"GIF87a": "GIF",
            b"GIF89a": "GIF",
            b"RIFF": "WEBP",
        }
        is_valid = False
        for magic, fmt in valid_formats.items():
            if content.startswith(magic):
                is_valid = True
                print(f"[image] detected format: {fmt} from {image_url}")
                break

        if not is_valid:
            print(f"[image] warning: {image_url} may not be a valid image format")

        return base64.b64encode(content).decode("utf-8")
    except Exception as e:
        print(f"[image] error loading from URL {image_url}: {e}")
        return None


def _clean_base64(image_b64: str) -> Optional[str]:
    import base64
    import re

    if image_b64.startswith("data:"):
        match = re.match(r"data:image/[^;]+;base64,(.+)", image_b64)
        if match:
            image_b64 = match.group(1)
        else:
            return None

    image_b64 = image_b64.strip()

    try:
        if not re.match(r"^[A-Za-z0-9+/=]+$", image_b64):
            print("[image] invalid base64 characters")
            return None

        decoded = base64.b64decode(image_b64, validate=True)

        if len(decoded) == 0:
            print("[image] empty image data")
            return None

        if len(decoded) < 10:
            print(f"[image] image too small ({len(decoded)} bytes)")
            return None

        return image_b64
    except Exception as e:
        print(f"[image] base64 validation error: {e}")
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# def _filter_results_by_score_and_delta(
#     results: List[Dict[str, Any]],
#     threshold_primary: float = 0.80,
#     threshold_fallback: float = 0.70,
#     max_delta: float = 0.07,
# ) -> List[Dict[str, Any]]:
#    def select(threshold: float) -> List[Dict[str, Any]]:
#        selected: List[Dict[str, Any]] = []
#        prev_score: Optional[float] = None

#        for item in results:
#            score = _as_float(item.get("bm25_score"))
#            if score is None or score < threshold:
#                continue
#            if prev_score is not None:
#                delta = abs(score - prev_score)
#                if delta > max_delta:
#                    continue
#            selected.append(item)
#            prev_score = score
#        return selected

#    first_pass = select(threshold_primary)
#    if first_pass:
#        return first_pass

#    second_pass = select(threshold_fallback)
#    if second_pass:
#        return second_pass

#    return results[:1]


def _vertex_embed(
    image_b64: Optional[str] = None,
    text: Optional[str] = None,
    model: str = "multimodalembedding@001",
):
    if not _VERTEX_AVAILABLE:
        raise RuntimeError("google-cloud-aiplatform not installed")
    project = _discover_gcp_project()
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "Cannot determine GCP project_id from credentials; set GOOGLE_APPLICATION_CREDENTIALS(_JSON)."
        )
    _ensure_gcp_adc()
    from vertexai.vision_models import MultiModalEmbeddingModel, Image

    mdl = MultiModalEmbeddingModel.from_pretrained(model)
    import base64

    image = None
    if image_b64:
        image_bytes = base64.b64decode(image_b64)
        image = Image(image_bytes)
    resp = mdl.get_embeddings(image=image, contextual_text=text)
    if getattr(resp, "image_embedding", None):
        return list(resp.image_embedding)
    if getattr(resp, "text_embedding", None):
        return list(resp.text_embedding)
    if getattr(resp, "embedding", None):
        return list(resp.embedding)
    raise RuntimeError("No embedding returned from Vertex AI")


def describe_mechanical_part(image_b64: str) -> str:
    """Caption v7b — discriminazione forte per tipo componente, blocco/corpo aggiunto. Retry su 429."""
    if _OPENAI_CLIENT is None:
        return ""
    for attempt in range(5):
        try:
            resp = _OPENAI_CLIENT.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0,
                max_tokens=500,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sei un esperto di disegno meccanico e information retrieval. "
                            "Riceverai immagini di tavole tecniche con pezzi meccanici. "
                            "Devi produrre una descrizione ottimizzata per ricerca ibrida (BM25 + vettoriale) "
                            "per trovare pezzi meccanici geometricamente simili.\n\n"
                            "OBIETTIVO CRITICO: pezzi di TIPO diverso devono produrre descrizioni MOLTO diverse. "
                            "Una flangia e un albero sono componenti radicalmente diversi per topologia, proporzioni e funzione. "
                            "Un blocco fresato e un raccordo tornito sono componenti radicalmente diversi. "
                            "La tua descrizione deve catturare queste differenze in modo netto.\n\n"
                            "REGOLE ASSOLUTE — violarne una invalida l'intera risposta:\n"
                            "1. NON leggere MAI testo, sigle, codici, numeri o quote dal disegno. "
                            "Ignora cartiglio, intestazioni, note, simboli di quotatura, tolleranze, annotazioni. "
                            "Questo include sigle come M5, M6, M10, G 1/4, Tr16, ecc. — NON menzionarle MAI.\n"
                            "2. NON includere quote numeriche (niente mm, diametri, raggi, lunghezze). "
                            "Unica eccezione: angoli chiaramente visibili dalla geometria (es. 45°, 30°).\n"
                            "3. NON specificare il tipo di filettatura (gas, metrica, conica, cilindrica, BSP, NPT). "
                            "Scrivi solo 'filettatura interna' o 'filettatura esterna'.\n"
                            "4. NON inventare elementi che non vedi chiaramente. "
                            "Se non sei sicuro che un elemento esista, NON menzionarlo. "
                            "Meglio omettere che allucinare.\n"
                            "5. Descrivi TUTTE le parti visibili del pezzo, comprese appendici, bracci, ganasce, "
                            "orecchie, staffe, alette e qualsiasi elemento non assial-simmetrico.\n"
                            "6. Se ci sono viste isometriche o 3D, usale per capire la forma complessiva "
                            "prima di descrivere i dettagli dalle sezioni.\n"
                            "7. QUANTIFICA sempre: conta i fori, i gradini, i diametri diversi, le gole. "
                            "Scrivi 'due fori passanti', 'tre gradini', 'quattro fori su corona circolare', non 'fori' generico.\n"
                            "8. DESCRIVI la disposizione dei fori: su corona circolare, allineati, singolo centrale, ecc.\n"
                            "9. Scrivi SOLO affermazioni certe. NON usare mai: 'presumibilmente', 'probabilmente', "
                            "'potrebbe', 'sembra', 'apparentemente', 'possibile'. Se non sei sicuro, ometti.\n"
                            "10. La PRIMA FRASE deve identificare il TIPO e la MACRO-TOPOLOGIA del pezzo. "
                            "Usa esattamente questo formato: '[TIPO] — [topologia].' Esempi:\n"
                            "   - 'Flangia (disco forato) — corpo piatto a disco con corona di fori.'\n"
                            "   - 'Albero (shaft) — corpo cilindrico allungato sviluppato in lunghezza.'\n"
                            "   - 'Raccordo (nipplo) — corpo tozzo e compatto tornito con cavità passante.'\n"
                            "   - 'Staffa (supporto) — piastra sagomata con bracci e fori di fissaggio.'\n"
                            "   - 'Boccola (manicotto) — cilindro cavo corto con pareti spesse.'\n"
                            "   - 'Blocco (corpo supporto/housing) — solido massiccio fresato dal pieno con foro centrale e fori di fissaggio.'\n"
                            "La macro-topologia deve distinguere chiaramente:\n"
                            "   - DISCO/PIATTO (flangia, piastra): sviluppo radiale >> assiale\n"
                            "   - ALLUNGATO (albero, perno, asta): sviluppo assiale >> radiale\n"
                            "   - TOZZO/COMPATTO (raccordo, boccola, dado): proporzioni equilibrate, TORNITO\n"
                            "   - MASSICCIO/BLOCCO (corpo, housing, blocco valvola): solido squadrato/cubico FRESATO dal pieno, "
                            "con fori e cavità ricavate per asportazione. NON confondere con raccordi torniti.\n"
                            "   - SAGOMATO (staffa, ganascia, supporto): geometria non assial-simmetrica\n\n"
                            "ATTENZIONE: se il pezzo ha forma squadrata/cubica/rettangolare con fori ricavati "
                            "dal pieno e smussi sugli spigoli, è un BLOCCO (corpo/housing), NON un raccordo."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Osserva l'immagine e analizza la geometria del pezzo. "
                                    "Usa TUTTE le viste disponibili (frontale, laterale, sezione, dettagli, isometrica).\n\n"
                                    "Scrivi una descrizione continua (NO elenchi numerati) che copra, IN QUESTO ORDINE:\n\n"
                                    "1. TIPO + MACRO-TOPOLOGIA (PRIMA FRASE OBBLIGATORIA) — "
                                    "identifica il pezzo e la sua topologia dominante. "
                                    "Formato: '[Tipo] ([sinonimo]) — [topologia].' "
                                    "Scegli il tipo più specifico tra: "
                                    "nipplo, raccordo, flangia, albero, boccola, puleggia, perno, dado, distanziale, "
                                    "manicotto, ghiera, supporto, staffa, piastra, tappo, adattatore, corpo valvola, "
                                    "ganascia, morsetto, collettore, disco, anello, asta, tubo, "
                                    "blocco, corpo, housing, basamento.\n\n"
                                    "2. PROPORZIONI E LAVORAZIONE — "
                                    "rapporto dimensionale dominante (diametro >> lunghezza = disco/flangia; "
                                    "lunghezza >> diametro = albero/perno; bilanciato = raccordo/boccola; "
                                    "squadrato/cubico = blocco/corpo) e "
                                    "categoria di lavorazione (tornito, fresato dal pieno, lamiera piegata/saldata, "
                                    "fusione, ricavato dal pieno).\n\n"
                                    "3. FORMA COMPLESSIVA — parti che formano il pezzo. "
                                    "Se ha appendici, bracci, ganasce, alette o parti non simmetriche, descrivile. "
                                    "Inizia dalla forma generale prima dei dettagli.\n\n"
                                    "4. PROFILO ESTERNO — simmetria, sezioni cilindriche/esagonali/coniche/squadrate, "
                                    "gradini (spalle) contandoli, smussi (chamfer), raccordi, flange.\n\n"
                                    "5. CAVITÀ E FORI — fori passanti/ciechi contandoli e descrivendo la disposizione "
                                    "(corona circolare, allineati, singolo centrale, agli angoli), filettature interne, "
                                    "gole anulari (sedi O-ring), gradini interni contandoli, "
                                    "cave per chiavetta (linguetta), scanalature.\n\n"
                                    "6. FUNZIONE OSSERVABILE — sedi di tenuta, zone filettate, "
                                    "riduzione diametro (riduttore), bloccaggio, accoppiamento, "
                                    "trasmissione coppia (scanalature, chiavette), alloggiamento (housing).\n\n"
                                    "Regole:\n"
                                    "- La PRIMA FRASE determina la categoria del pezzo — sceglila con cura.\n"
                                    "- Se il pezzo è squadrato/cubico con fori ricavati, è un BLOCCO non un raccordo.\n"
                                    "- Menziona SOLO elementi che vedi chiaramente. Nel dubbio, ometti.\n"
                                    "- QUANTIFICA: conta fori, gradini, diametri diversi, gole, denti.\n"
                                    "- NON includere sigle di filettatura (M5, M6, G1/4, ecc.).\n"
                                    "- NON usare 'presumibilmente', 'probabilmente', 'potrebbe', 'sembra'.\n"
                                    "- Lessico meccanico con sinonimi: gola anulare (sede O-ring), smusso (chamfer), "
                                    "gradino (spalla), raccordo (raggio di raccordo), cava per chiavetta (linguetta).\n"
                                    "- Massimo 6 frasi, massimo 1000 caratteri.\n"
                                    "- Italiano tecnico, testo continuo."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    },
                ],
            )
            caption = (resp.choices[0].message.content or "").strip()
            return caption[:1200] if len(caption) > 1200 else caption
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg:
                wait = 2 ** attempt
                print(f"  429 retry {attempt+1}/5 — sleep {wait}s")
                time.sleep(wait)
            else:
                print(f"WARN caption: {e}")
                return ""
    print("WARN caption: max retry raggiunto")
    return ""


def _extract_dim_values_for_query(image_b64: str) -> str:
    """
    Estrae le dimensioni principali e le normalizza rispetto al massimo.
    [100, 50, 20] → '1.0, 0.5, 0.2'  (scale-invariant, compatibile con dim_values in Sinde5)
    """
    if _OPENAI_CLIENT is None:
        return ""
    try:
        resp = _OPENAI_CLIENT.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            max_tokens=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei un esperto di disegno meccanico. "
                        "Identifica le dimensioni che definiscono la forma geometrica principale del pezzo. "
                        "Ignora tolleranze, filettature, smussi, raggi e quote secondarie. "
                        "Restituisci SOLO i valori numerici in ordine decrescente, separati da virgola, "
                        "senza unita di misura. Massimo 6 valori. "
                        "Se non vedi quote leggibili, restituisci stringa vuota."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Dimensioni principali della forma geometrica, ordine decrescente, solo numeri separati da virgola.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return ""
        nums = [float(x.replace(",", ".")) for x in re.findall(r"[\d]+(?:[.,]\d+)?", raw)]
        nums = sorted([v for v in nums if v > 0], reverse=True)[:6]
        if not nums:
            return ""
        mx = nums[0]
        return ", ".join(str(round(v / mx, 4)) for v in nums)
    except Exception as e:
        print(f"[query-caption] errore extract_dim_values: {e}")
        return ""


def _parse_normalized_dims(dim_values_str: str) -> List[float]:
    """Parsa '1.0, 0.5, 0.2' → [1.0, 0.5, 0.2]. Valori già normalizzati."""
    if not dim_values_str:
        return []
    try:
        return [float(x.strip()) for x in dim_values_str.split(",") if x.strip()]
    except Exception:
        return []


_DIMS_DELIMITER = "---DIMS---"
_MAX_SHAPE_CAPTION_CHARS = 1000
_MAX_DIM_CAPTION_CHARS = 400
_MAX_COMBINED_CAPTION_CHARS = 1200


@dataclass
class MechanicalDrawingCaption:
    shape_caption: str
    dim_caption: str
    combined: str


def _extract_numeric_dims(text: str) -> List[float]:
    """
    Estrae tutti i valori numerici da una stringa di dimensioni (dim_caption o
    sezione dopo ---DIMS--- nel caption). Restituisce lista ordinata decrescente.
    I valori di filettatura (M, G, UNC...) vengono ignorati — contano solo misure lineari/radiali.
    """
    # Rimuove token di filettatura (M10x1, G 3/8-19, UNC...) per non inquinare i rapporti
    cleaned = re.sub(r"\b(?:M|G|UNC|BSP|BSW|NPT)\s*[\d./\-x×]+[^\s,]*", "", text, flags=re.IGNORECASE)
    # Estrae tutti i float dopo Ø/⌀ o standalone
    values = [float(m) for m in re.findall(r"[\d]+(?:[.,]\d+)?", cleaned.replace(",", "."))]
    # Filtra valori irrilevanti (<1 o >2000) e deduplicati
    values = sorted({v for v in values if 1.0 <= v <= 2000.0}, reverse=True)
    return values


def _proportional_dim_similarity(query_dims: List[float], result_dims: List[float]) -> float:
    """
    Similarità proporzionale scale-invariant tra due set di dimensioni.
    Normalizza entrambi per il loro max (→ vettori in [0,1]) e calcola
    la cosine similarity sul numero di componenti condivise.
    Un rettangolo 2×5 e uno 12×30 → score ~1.0; un 5×2 vs 2×5 → score ~0.78.
    Restituisce 1.0 se uno dei due set è vuoto (nessuna penalità senza dati).
    """
    if not query_dims or not result_dims:
        return 1.0

    def normalize(dims: List[float]) -> List[float]:
        mx = dims[0]  # già ordinati desc
        return [v / mx for v in dims]

    q = normalize(query_dims)
    r = normalize(result_dims)

    # Allinea alla lunghezza minima (confronta solo le prime N dimensioni comuni)
    n = min(len(q), len(r), 6)
    q, r = q[:n], r[:n]

    dot = sum(a * b for a, b in zip(q, r))
    norm_q = sum(a * a for a in q) ** 0.5
    norm_r = sum(b * b for b in r) ** 0.5
    if norm_q == 0 or norm_r == 0:
        return 1.0
    return dot / (norm_q * norm_r)


def _rerank_by_proportional_dims(
    results: List[Dict], query_dim_values: str, weight: float = 0.35
) -> List[Dict]:
    """
    Re-ordina i risultati ibridi combinando lo score Weaviate con la similarità
    proporzionale delle dimensioni. weight=0 disabilita il re-ranking.
    query_dim_values: stringa normalizzata '1.0, 0.5, 0.2' (dal campo dim_values di Sinde5).
    """
    if weight <= 0 or not query_dim_values:
        return results

    query_dims = _parse_normalized_dims(query_dim_values)
    if not query_dims:
        return results

    for item in results:
        dim_values_str = (item.get("properties") or {}).get("dim_values", "") or ""
        result_dims = _parse_normalized_dims(dim_values_str)
        dim_sim = _proportional_dim_similarity(query_dims, result_dims)
        base = item.get("bm25_score") or 0.0
        item["dim_similarity"] = round(dim_sim, 4)
        item["combined_score"] = round(base * (1.0 - weight + weight * dim_sim), 4)

    results.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return results


def _normalize_caption_for_search(text: str) -> str:
    """
    Generalizza dimensioni assolute nel testo shape per ridurre mismatch BM25
    su valori numerici specifici (es. Ø90 vs Ø120 tra pezzi simili).
    """
    if not text:
        return text

    result = text
    result = re.sub(
        r"\bM\d+(?:[×x]\d+(?:[.,]\d+)?)?\b",
        "filettatura metrica",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"(?:Ø|⌀|phi\.?)\s*[\d.,]+",
        "diametro",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"\d+(?:[.,]\d+)?\s*mm\b",
        "dimensione mm",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\bR\s*[\d.,]+\b", "raggio", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def _build_mechanical_drawing_combined(
    shape_caption: str, dim_caption: str
) -> str:
    shape = shape_caption.strip()
    dims = dim_caption.strip()
    if shape and dims:
        combined = f"{shape}\n\n{_DIMS_DELIMITER}\n{dims}"
    else:
        combined = shape or dims
    if len(combined) > _MAX_COMBINED_CAPTION_CHARS:
        combined = combined[:_MAX_COMBINED_CAPTION_CHARS]
    return combined


def _parse_mechanical_drawing_response(raw: str) -> MechanicalDrawingCaption:
    shape_caption = ""
    dim_caption = ""

    text = (raw or "").strip()
    if not text:
        return MechanicalDrawingCaption("", "", "")

    if _DIMS_DELIMITER in text:
        parts = text.split(_DIMS_DELIMITER, 1)
        shape_caption = parts[0].strip()
        dim_caption = parts[1].strip() if len(parts) > 1 else ""
    else:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                shape_caption = str(parsed.get("shape_caption") or "").strip()
                dim_caption = str(parsed.get("dim_caption") or "").strip()
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict):
                        shape_caption = str(parsed.get("shape_caption") or "").strip()
                        dim_caption = str(parsed.get("dim_caption") or "").strip()
                except json.JSONDecodeError:
                    pass
        if not shape_caption and not dim_caption:
            shape_caption = text

    if len(shape_caption) > _MAX_SHAPE_CAPTION_CHARS:
        shape_caption = shape_caption[:_MAX_SHAPE_CAPTION_CHARS]
    if len(dim_caption) > _MAX_DIM_CAPTION_CHARS:
        dim_caption = dim_caption[:_MAX_DIM_CAPTION_CHARS]

    combined = _build_mechanical_drawing_combined(shape_caption, dim_caption)
    return MechanicalDrawingCaption(
        shape_caption=shape_caption,
        dim_caption=dim_caption,
        combined=combined,
    )


def describe_mechanical_drawing(
    image_b64: str, mode: str = "query"
) -> Optional[MechanicalDrawingCaption]:
    """Wrapper retrocompatibile — delega a describe_mechanical_part."""
    caption = describe_mechanical_part(image_b64)
    if not caption:
        return None
    return MechanicalDrawingCaption(shape_caption=caption, dim_caption="", combined=caption)


def describe_image_for_query(image_b64: str) -> Optional[str]:
    """Wrapper retrocompatibile: usa describe_mechanical_part."""
    caption = describe_mechanical_part(image_b64)
    return caption or None


# =====================================================
# DINOv2 + Fusion Search (DINOv2 image + Caption hybrid)
# =====================================================

_DINO_COLLECTION_NAME = os.environ.get("DINO_COLLECTION", "SindeDino")
_FUSION_W_DINO = float(os.environ.get("FUSION_W_DINO", "0.5"))
_FUSION_W_CAP = float(os.environ.get("FUSION_W_CAP", "0.5"))
_FUSION_POOL = int(os.environ.get("FUSION_POOL", "50"))
_FUSION_ENABLED = os.environ.get("FUSION_ENABLED", "1").lower() in ("1", "true", "yes")

_dino_model = None
_dino_processor = None
_dino_device = None


def _ensure_dino_loaded():
    """Lazy-load DINOv2 model on first use."""
    global _dino_model, _dino_processor, _dino_device
    if _dino_model is not None:
        return True
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        _dino_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "facebook/dinov2-base"
        print(f"[dino] Loading {model_name} on {_dino_device}...")
        _dino_processor = AutoImageProcessor.from_pretrained(model_name)
        _dino_model = AutoModel.from_pretrained(model_name).to(_dino_device).eval()
        print(f"[dino] Model loaded (768 dim)")
        return True
    except Exception as exc:
        print(f"[dino] Failed to load DINOv2: {exc}")
        return False


def _preprocess_drawing(pil_image):
    """Crop cartiglio, Otsu binarization, connected component filtering, morph close."""
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img.copy()
    h, w = gray.shape

    gray = gray[: int(h * 0.82), : int(w * 0.95)]

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 3000:
            clean[labels == i] = 255

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    result_rgb = cv2.cvtColor(255 - clean, cv2.COLOR_GRAY2RGB)
    return PILImage.fromarray(result_rgb)


def _embed_image_dino(pil_image):
    """Preprocess + DINOv2 CLS embedding (768-dim, L2-normalized)."""
    import torch

    _ensure_dino_loaded()
    clean = _preprocess_drawing(pil_image)
    inputs = _dino_processor(images=clean, return_tensors="pt").to(_dino_device)
    with torch.no_grad():
        outputs = _dino_model(**inputs)
    cls = outputs.last_hidden_state[:, 0]
    cls = cls / cls.norm(dim=-1, keepdim=True)
    return cls[0].cpu().tolist()


def _fusion_normalize(results, score_key="score"):
    """Min-max normalize scores to [0, 1]."""
    scores = [r.get(score_key) or 0 for r in results]
    if not scores:
        return results
    mn, mx = min(scores), max(scores)
    rng = mx - mn if mx > mn else 1.0
    return [{**r, "norm_score": ((r.get(score_key) or 0) - mn) / rng} for r in results]


def _do_fusion_search(client, image_b64, limit=20):
    """
    Fusion DINOv2 + Caption: fetches top-pool from both SindeDino (near_vector)
    and Sinde5 (hybrid caption), normalizes scores, combines with weighted sum.
    Returns dict compatible with hybrid_search output format.
    """
    from PIL import Image as PILImage
    from io import BytesIO

    pool = _FUSION_POOL
    w_dino = _FUSION_W_DINO
    w_cap = _FUSION_W_CAP

    # 1. DINOv2 embedding
    img_bytes = base64.b64decode(image_b64)
    pil_img = PILImage.open(BytesIO(img_bytes)).convert("RGB")
    query_embedding = _embed_image_dino(pil_img)

    # 2. DINOv2 near_vector on SindeDino
    dino_coll = client.collections.get(_DINO_COLLECTION_NAME)
    resp_dino = dino_coll.query.near_vector(
        near_vector=query_embedding,
        limit=pool,
        return_properties=["source_pdf", "image_b64"],
        return_metadata=MetadataQuery(distance=True),
    )
    res_dino = [
        {
            "source_pdf": o.properties.get("source_pdf", ""),
            "score": 1.0 - (getattr(o.metadata, "distance", 0) or 0),
            "image_b64": o.properties.get("image_b64"),
        }
        for o in getattr(resp_dino, "objects", [])
    ]

    # 3. Caption hybrid on Sinde5 (skip if w_cap == 0 → pure DINOv2 mode)
    res_cap = []
    if w_cap > 0:
        caption = describe_mechanical_part(image_b64) or ""
        if caption:
            print(f"[fusion] caption: {caption[:100]}...")

        _update_client_grpc_metadata(client)
        cap_coll_name = _get_default_collection()
        cap_coll = client.collections.get(cap_coll_name)

        if caption:
            try:
                resp_cap = cap_coll.query.hybrid(
                    query=caption,
                    alpha=0.2,
                    query_properties=["caption"],
                    limit=pool,
                    return_properties=["source_pdf", "caption", "image_b64"],
                    return_metadata=MetadataQuery(score=True),
                )
            except Exception as exc:
                print(f"[fusion] hybrid caption failed, BM25 fallback: {exc}")
                resp_cap = cap_coll.query.bm25(
                    query=caption,
                    query_properties=["caption"],
                    limit=pool,
                    return_properties=["source_pdf", "caption", "image_b64"],
                    return_metadata=MetadataQuery(score=True),
                )
            res_cap = [
                {
                    "source_pdf": o.properties.get("source_pdf", ""),
                    "score": getattr(o.metadata, "score", None),
                    "properties": o.properties,
                }
                for o in getattr(resp_cap, "objects", [])
            ]
    else:
        print("[fusion] DINOv2-only mode (w_cap=0)")

    # 4. Normalize
    res_dino_n = _fusion_normalize(res_dino, "score")
    res_cap_n = _fusion_normalize(res_cap, "score")

    dino_map = {r["source_pdf"]: r["norm_score"] for r in res_dino_n}
    dino_img_map = {r["source_pdf"]: r.get("image_b64") for r in res_dino}

    cap_map = {r["source_pdf"]: r["norm_score"] for r in res_cap_n}
    cap_props_map = {r["source_pdf"]: r.get("properties", {}) for r in res_cap}

    # 6. Fuse
    all_pdfs = set(dino_map.keys()) | set(cap_map.keys())
    fused = []
    for pdf in all_pdfs:
        d = dino_map.get(pdf, 0.0)
        c = cap_map.get(pdf, 0.0)
        fused.append({
            "source_pdf": pdf,
            "fusion_score": w_dino * d + w_cap * c,
            "dino_norm": d,
            "cap_norm": c,
        })

    fused.sort(key=lambda x: x["fusion_score"], reverse=True)
    fused = fused[:limit]

    print(f"[fusion] w_dino={w_dino}, w_cap={w_cap}, pool={pool}, "
          f"dino_hits={len(dino_map)}, cap_hits={len(cap_map)}, fused={len(fused)}")

    # 7. Build output — prefer Sinde5 properties (name, mediaType, page_index)
    out = []
    for r in fused:
        pdf = r["source_pdf"]
        if pdf in cap_props_map and cap_props_map[pdf]:
            props = dict(cap_props_map[pdf])
        else:
            props = {
                "source_pdf": pdf,
                "name": pdf.replace(".pdf", ""),
                "image_b64": dino_img_map.get(pdf),
            }
        out.append({
            "uuid": "",
            "properties": props,
            "bm25_score": r["fusion_score"],
            "distance": None,
        })

    for i, r in enumerate(fused[:5]):
        print(f"[fusion] {i + 1}. {r['source_pdf']:35s} "
              f"f={r['fusion_score']:.4f} d={r['dino_norm']:.3f} c={r['cap_norm']:.3f}")

    return {"count": len(out), "results": out}


@mcp.tool()
def insert_image_vertex(
    collection: str,
    image_id: Optional[str] = None,
    image_url: Optional[str] = None,
    caption: Optional[str] = None,
    id: Optional[str] = None,
) -> Dict[str, Any]:
    image_b64 = None

    if image_id:
        if image_id in _UPLOADED_IMAGES:
            img_data = _UPLOADED_IMAGES[image_id]
            if img_data["expires_at"] > time.time():
                image_b64 = img_data["image_b64"]
            else:
                _UPLOADED_IMAGES.pop(image_id, None)
                return {
                    "error": f"Image ID {image_id} has expired. Please upload the image again."
                }
        else:
            return {
                "error": (
                    f"Image ID {image_id} not found. Use upload_image or /upload-image first."
                )
            }

    if image_url and not image_b64:
        image_b64 = _load_image_from_url(image_url)
        if not image_b64:
            return {"error": f"Failed to load image from URL: {image_url}"}
        image_b64 = _clean_base64(image_b64)
        if not image_b64:
            return {"error": f"Invalid image format from URL: {image_url}"}

    if not image_b64:
        return {"error": "Either image_id or image_url must be provided"}

    if caption is None:
        caption = describe_mechanical_part(image_b64) or None

    vec = _vertex_embed(image_b64=image_b64, text=caption)
    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}

        obj = coll.data.insert(
            properties={"caption": caption, "image_b64": image_b64},
            vectors={"image": vec},
        )
        return {
            "uuid": str(getattr(obj, "uuid", "")),
            "named_vector": "image",
        }
    finally:
        client.close()


@mcp.tool()
def image_search_vertex(
    collection: str,
    image_id: Optional[str] = None,
    image_url: Optional[str] = None,
    caption: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    # Usa la collection di default configurata, mantenendo lo stesso comportamento di forzatura
    default_collection = _get_default_collection()
    if not collection:
        collection = default_collection
    elif collection != default_collection:
        print(
            f"[image_search_vertex] warning: collection '{collection}' requested, "
            f"but using '{default_collection}' as per instructions"
        )
        collection = default_collection

    image_b64 = None

    if image_id:
        if image_id in _UPLOADED_IMAGES:
            img_data = _UPLOADED_IMAGES[image_id]
            if img_data["expires_at"] > time.time():
                image_b64 = img_data["image_b64"]
            else:
                _UPLOADED_IMAGES.pop(image_id, None)
                return {
                    "error": f"Image ID {image_id} has expired. Please upload the image again."
                }
        else:
            return {
                "error": (
                    f"Image ID {image_id} not found. Please upload the image first using upload_image."
                )
            }

    if image_url and not image_b64:
        image_b64 = _load_image_from_url(image_url)
        if not image_b64:
            return {"error": f"Failed to load image from URL: {image_url}"}
        image_b64 = _clean_base64(image_b64)
        if not image_b64:
            return {"error": f"Invalid image format from URL: {image_url}"}

    if not image_b64:
        return {"error": "Either image_id or image_url must be provided"}

    client = _connect()
    try:
        coll = client.collections.get(collection)
        if coll is None:
            return {"error": f"Collection '{collection}' not found"}
        resp = coll.query.near_image(
            image_b64,
            limit=limit,
            return_properties=["source_pdf", "caption", "image_b64"],
            return_metadata=MetadataQuery(distance=True),
        )
        out = []
        for o in getattr(resp, "objects", []) or []:
            out.append(
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", {}),
                    "distance": getattr(getattr(o, "metadata", None), "distance", None),
                }
            )
        return {"count": len(out), "results": out}
    finally:
        client.close()


@mcp.tool()
def diagnose_vertex() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["project_id"] = _discover_gcp_project()
    info["oauth_enabled"] = os.environ.get("VERTEX_USE_OAUTH", "").lower() in (
        "1",
        "true",
        "yes",
    )
    info["headers_active"] = bool(_VERTEX_HEADERS) if "_VERTEX_HEADERS" in globals() else False
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request

        SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
        gac_path = _resolve_service_account_path()
        token_preview = None
        expiry = None
        if gac_path and os.path.exists(gac_path):
            creds = service_account.Credentials.from_service_account_file(
                gac_path, scopes=SCOPES
            )
            creds.refresh(Request())
            token_preview = (creds.token[:12] + "...") if creds.token else None
            expiry = getattr(creds, "expiry", None)
        info["token_sample"] = token_preview
        info["token_expiry"] = str(expiry) if expiry else None
    except Exception as e:
        info["token_error"] = str(e)
    return info


# Registry dei tool normali che vuoi esporre alla App
TOOL_REGISTRY: Dict[str, Any] = {
    "get_instructions": get_instructions,
    "reload_instructions": reload_instructions,
    "get_config": get_config,
    "debug_widget": debug_widget,
    "check_connection": check_connection,
    "upload_image": upload_image,
    "list_collections": list_collections,
    "get_schema": get_schema,
    "keyword_search": keyword_search,
    "semantic_search": semantic_search,
    "hybrid_search": hybrid_search,
    "insert_image_vertex": insert_image_vertex,
    "image_search_vertex": image_search_vertex,  # Nota: questa non ha @mcp.tool() ma è una funzione normale
    "diagnose_vertex": diagnose_vertex,
    "get_last_sinde_results": get_last_sinde_results,
    # (opzionale) tieni ancora l'helper interno, ma NON serve come tool:
    # "sinde_widget_push_results": sinde_widget_push_results,
}

# Tool nascosti (non esposti all'LLM ma ancora disponibili internamente)
_HIDDEN_TOOLS: set[str] = {
    "debug_widget",
    "upload_image",
    "semantic_search",
    "keyword_search",
    "insert_image_vertex",
    "image_search_vertex",
    "diagnose_vertex",
}


# ==== Vertex OAuth Token Refresher (optional) ===============================
def _write_adc_from_json_env():
    gac_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    gac_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_json and not gac_path:
        tmp_path = "/app/gcp_credentials.json"
        with open(tmp_path, "w", encoding="utf-8") as f2:
            f2.write(gac_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_path
    _resolve_service_account_path()


def _refresh_vertex_oauth_loop():
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    import datetime
    import time

    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
    cred_path = _resolve_service_account_path()
    if not cred_path or not os.path.exists(cred_path):
        print("[vertex-oauth] GOOGLE_APPLICATION_CREDENTIALS missing; token refresher disabled")
        return
    creds = service_account.Credentials.from_service_account_file(
        cred_path, scopes=SCOPES
    )
    global _VERTEX_HEADERS
    while True:
        try:
            creds.refresh(Request())
            token = creds.token
            _VERTEX_HEADERS = _build_vertex_header_map(token)
            # Aggiorna anche le variabili d'ambiente per Weaviate vectorizer
            os.environ["GOOGLE_APIKEY"] = token
            os.environ["PALM_APIKEY"] = token
            token_preview = token[:10] if token else None
            print(f"[vertex-oauth] 🔄 Vertex token refreshed (prefix: {token_preview}...)")
            sleep_s = 55 * 60
            if creds.expiry:
                from datetime import timezone
                now = datetime.datetime.now(timezone.utc).replace(tzinfo=creds.expiry.tzinfo)
                delta = (creds.expiry - now).total_seconds() - 300
                if delta > 300:
                    sleep_s = int(delta)
            time.sleep(sleep_s)
        except Exception as e:
            print(f"[vertex-oauth] refresh error: {e}")
            time.sleep(60)


def _maybe_start_vertex_oauth_refresher():
    global _VERTEX_REFRESH_THREAD_STARTED
    if _VERTEX_REFRESH_THREAD_STARTED:
        return
    if os.environ.get("VERTEX_USE_OAUTH", "").lower() not in ("1", "true", "yes"):
        return
    _write_adc_from_json_env()
    sa_path = _resolve_service_account_path()
    if not sa_path:
        print("[vertex-oauth] service account path not found; refresher not started")
        return
    import threading

    t = threading.Thread(target=_refresh_vertex_oauth_loop, daemon=True)
    t.start()
    _VERTEX_REFRESH_THREAD_STARTED = True


_maybe_start_vertex_oauth_refresher()

# --- Alias /mcp senza slash finale, se serve --------------------------------
try:
    from starlette.routing import Route

    _starlette_app = getattr(mcp, "app", None) or getattr(mcp, "_app", None)

    if _starlette_app is not None:

        async def _mcp_alias(request):
            scope = dict(request.scope)
            scope["path"] = "/mcp/"
            scope["raw_path"] = b"/mcp/"
            return await _starlette_app(scope, request.receive, request.send)

        _starlette_app.router.routes.insert(
            0,
            Route(
                "/mcp",
                endpoint=_mcp_alias,
                methods=["GET", "HEAD", "POST", "OPTIONS"],
            ),
        )
except Exception as _route_err:
    print("[mcp] warning: cannot register MCP alias route:", _route_err)

# ==== Definizioni MCP di basso livello per il widget (come in Pizzaz) ====
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},          # nessun argomento per aprire il widget
    "required": [],
    "additionalProperties": False,
}


@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    """Espone il tool widget + tutti i tool normali a ChatGPT."""
    tools: List[types.Tool] = []

    # 1) Tool del widget (quello con la UI)
    w = SINDE_WIDGET
    tools.append(
        types.Tool(
            name=w.identifier,
            title=w.title,
            description=w.title,
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            _meta=_tool_meta(w),  # <<< QUI sta openai/outputTemplate ecc.
            annotations={
                "destructiveHint": False,
                "openWorldHint": False,
                "readOnlyHint": True,
            },
        )
    )

    # 2) Tutti gli altri tool normali (escludendo quelli nascosti)
    for name in TOOL_REGISTRY.keys():
        # Salta i tool nascosti
        if name in _HIDDEN_TOOLS:
            continue
        # Schema di default: argomenti liberi
        input_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        }

        tool_title = name
        tool_description = name
        annotations = {
            "destructiveHint": False,
            "openWorldHint": True,
            "readOnlyHint": False,
        }

        # ✅ Tool speciale per recuperare i risultati dal widget Sinde
        if name == "get_last_sinde_results":
            input_schema = {
                "type": "object",
                "properties": {},
                "required": [],               # nessun argomento richiesto
                "additionalProperties": False,
            }
            tool_title = "Risultati ricerca immagini Sinde"
            tool_description = (
                "Recupera gli ultimi risultati della ricerca immagini mostrati nel widget Sinde. "
                "Usalo automaticamente quando l'utente parla dei 'risultati del widget', "
                "'primo risultato', 'secondo risultato', 'riassumi i risultati della ricerca immagini', ecc."
            )
            annotations = {
                "destructiveHint": False,
                "openWorldHint": False,
                "readOnlyHint": True,
            }

        # ✅ Schema specifico per hybrid_search con istruzioni incluse
        elif name == "hybrid_search":
            input_schema = {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Nome della collection (sempre 'Sinde5' per questo assistente)",
                    },
                    "query": {
                        "type": "string",
                        "description": "Query di ricerca testuale",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Numero massimo di risultati da restituire",
                        "default": 20,
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Peso della ricerca vettoriale (0.0 = solo keyword, 1.0 = solo vettoriale). Default configurabile con HYBRID_DEFAULT_ALPHA (default 0.2).",
                        "default": _get_default_alpha(),
                    },
                    "query_properties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Proprietà su cui cercare (default: ['caption', 'name'])",
                    },
                    "return_properties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Proprietà da restituire (default: ['name', 'source_pdf', 'page_index', 'mediaType'])",
                    },
                    "image_id": {
                        "type": "string",
                        "description": "ID dell'immagine caricata tramite /upload-image",
                    },
                    "image_url": {
                        "type": "string",
                        "description": "URL pubblico dell'immagine da usare per la ricerca",
                    },
                },
                "required": ["collection", "query"],
                "additionalProperties": False,
            }
            tool_title = "Ricerca ibrida (BM25 + vettoriale)"
            tool_description = (
                "Esegue una ricerca ibrida combinando ricerca keyword (BM25) e ricerca vettoriale. "
                "Tool principale per cercare nella collection Sinde5.\n\n"
                "ISTRUZIONI: Usa SEMPRE collection='Sinde5'. Usa query_properties=['caption'] e "
                "return_properties=['source_pdf','caption','image_b64']. Mantieni alpha=0.2 e limit=20 "
                "salvo richieste diverse. Per ricerche per immagini, usa image_id (da /upload-image) o image_url."
            )

        tools.append(
            types.Tool(
                name=name,
                title=tool_title,
                description=tool_description,
                inputSchema=input_schema,
                annotations=annotations,
            )
        )

    return tools


@mcp._mcp_server.list_resources()
async def _list_resources() -> List[types.Resource]:
    w = SINDE_WIDGET
    return [
        types.Resource(
            name=w.title,
            title=w.title,
            uri=w.template_uri,
            description=_resource_description(w),
            mimeType=MIME_TYPE,
            _meta=_tool_meta(w),
        )
    ]


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> List[types.ResourceTemplate]:
    w = SINDE_WIDGET
    return [
        types.ResourceTemplate(
            name=w.title,
            title=w.title,
            uriTemplate=w.template_uri,
            description=_resource_description(w),
            mimeType=MIME_TYPE,
            _meta=_tool_meta(w),
        )
    ]


async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    w = SINDE_WIDGET
    if str(req.params.uri) != w.template_uri:
        return types.ServerResult(
            types.ReadResourceResult(
                contents=[],
                _meta={"error": f"Unknown resource: {req.params.uri}"},
            )
        )

    contents = [
        types.TextResourceContents(
            uri=w.template_uri,
            mimeType=MIME_TYPE,
            text=w.html,
            _meta=_tool_meta(w),
        )
    ]

    return types.ServerResult(types.ReadResourceResult(contents=contents))


async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    name = req.params.name
    args = req.params.arguments or {}

    # LOG DI DEBUG: vediamo quali tool vengono chiamati
    print(f"[call_tool] name={name}, args={json.dumps(args, ensure_ascii=False)}")

    # 1) Tool del widget (UI)
    if name == SINDE_WIDGET.identifier:
        w = SINDE_WIDGET
        meta = {
            "openai/toolInvocation/invoking": w.invoking,
            "openai/toolInvocation/invoked": w.invoked,
        }
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=w.response_text,
                    )
                ],
                structuredContent={"widgetReady": True},
                _meta=meta,
            )
        )

    # 2) Tool normali (quelli del registry)
    if name in TOOL_REGISTRY:
        fn = TOOL_REGISTRY[name]

        if name == "get_last_sinde_results":
            print("[call_tool] get_last_sinde_results invoked")

        # Caso speciale: hybrid_search → ripuliamo gli argomenti (niente return_properties)
        if name == "hybrid_search":
            print("[call_tool] hybrid_search called with:", args)

            clean_args: Dict[str, Any] = {}

            # collection con default preso da WEAVIATE_DEFAULT_COLLECTION (o 'Sinde3')
            clean_args["collection"] = args.get("collection") or _get_default_collection()

            # query obbligatoria
            q = args.get("query")
            if not q:
                return types.ServerResult(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text="Errore: parametro obbligatorio 'query' mancante per hybrid_search.",
                            )
                        ],
                        isError=True,
                    )
                )
            clean_args["query"] = q

            # parametri opzionali
            if "limit" in args:
                clean_args["limit"] = args["limit"]
            if "alpha" in args:
                clean_args["alpha"] = args["alpha"]
            if "query_properties" in args:
                clean_args["query_properties"] = args["query_properties"]
            if "image_id" in args:
                clean_args["image_id"] = args["image_id"]
            if "image_url" in args:
                clean_args["image_url"] = args["image_url"]

            # 🔴 QUI LA COSA IMPORTANTE:
            # sovrascriviamo args con la versione ripulita
            # (così return_properties e qualsiasi altro extra SPARISCONO)
            args = clean_args

        # Tutti gli altri tool normali rimangono come prima
        try:
            # Proviamo a passare gli argomenti così come sono
            result = fn(**args)
            # Se la funzione è async, await
            if hasattr(result, "__await__"):
                result = await result
        except TypeError as e:
            # Se la firma non combacia (ad es. tool senza parametri), riproviamo senza args
            try:
                result = fn()
                if hasattr(result, "__await__"):
                    result = await result
            except Exception as e2:
                return types.ServerResult(
                    types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"Errore chiamando tool {name}: {e2}",
                            )
                        ],
                        isError=True,
                    )
                )
        except Exception as e:
            return types.ServerResult(
                types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=f"Errore chiamando tool {name}: {e}",
                        )
                    ],
                    isError=True,
                )
            )

        # Testo diverso se è il tool del widget
        if name == "get_last_sinde_results":
            text_msg = (
                "Ho recuperato i risultati dal widget Sinde. "
                "Trovi il riassunto e i dati in structuredContent.summary e structuredContent.raw_results."
            )
        else:
            text_msg = f"Risultato del tool {name} disponibile in structuredContent."

        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=text_msg,
                    )
                ],
                structuredContent=(
                    result if isinstance(result, dict) else {"result": result}
                ),
            )
        )

    # 3) Tool sconosciuto
    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ],
            isError=True,
        )
    )


# Registra i request handler sul server MCP
mcp._mcp_server.request_handlers[types.CallToolRequest] = _call_tool_request
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource


# ==== Esponi l'app ASGI per uvicorn (per uso diretto nello start command) ====
# Puoi usare: uvicorn serve:app --host 0.0.0.0 --port $PORT
# Come nell'esempio Pizzaz, usiamo semplicemente mcp.streamable_http_app()
try:
    app = mcp.streamable_http_app()
    if app is None:
        raise ValueError("streamable_http_app() returned None")
    print("[mcp] app obtained via streamable_http_app()")
except Exception as e:
    print(f"[mcp] error getting app via streamable_http_app(): {e}")
    # Fallback: prova a ottenere l'app in altri modi
    from starlette.applications import Starlette
    app = None
    for attr_name in ["app", "_app", "asgi_app", "_asgi_app"]:
        app = getattr(mcp, attr_name, None)
        if app and isinstance(app, Starlette):
            print(f"[mcp] found app via mcp.{attr_name} (fallback)")
            break
    if app is None:
        raise RuntimeError("Cannot get FastMCP app - streamable_http_app() failed and no app found")


# Aggiungi CORS middleware se disponibile (opzionale)
try:
    from starlette.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
except Exception:
    pass

# ==== main: avvia il server con uvicorn (come nell'esempio Pizzaz) ==================
if __name__ == "__main__":
    import uvicorn
    
    # Porta/host che Render si aspetta
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "10000"))
    
    # Usa uvicorn direttamente con l'app esposta
    # Come nell'esempio Pizzaz: uvicorn.run("main:app", host="0.0.0.0", port=8000)
    uvicorn.run("serve:app", host=host, port=port)

