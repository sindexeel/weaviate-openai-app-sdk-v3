"""
PoC Phase 0 — Generazione codice CAD parametrico da immagini di disegni tecnici.

Uso da riga di comando:
    python poc_phase0.py path/to/drawing.pdf
    python poc_phase0.py path/to/drawing.png

Richiede:
    OPENAI_API_KEY nel environment (o in un file .env nella stessa cartella)
"""

import ast
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Caricamento .env (se presente)
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv()


# ---------------------------------------------------------------------------
# Conversione PDF → PNG base64
# ---------------------------------------------------------------------------

def pdf_to_png_base64(pdf_path: str) -> Optional[str]:
    """Converte la prima pagina di un PDF in PNG base64 a 300 DPI."""
    try:
        import pypdfium2 as pdfium
        from PIL import Image  # noqa: F401
    except ImportError as e:
        print(f"[pdf] dipendenza mancante: {e}")
        return None

    try:
        data = Path(pdf_path).read_bytes()
        pdf = pdfium.PdfDocument(data)
        if len(pdf) == 0:
            print("[pdf] documento senza pagine")
            return None
        page = pdf[0]
        bitmap = page.render(scale=3).to_pil()  # scale=3 ≈ 300 DPI
        buf = BytesIO()
        bitmap.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[pdf] errore conversione: {e}")
        return None


def png_to_base64(png_path: str) -> Optional[str]:
    """Legge un PNG da disco e lo restituisce come base64."""
    try:
        data = Path(png_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        print(f"[png] errore lettura: {e}")
        return None


def image_path_to_base64(path: str) -> Optional[str]:
    """Dispatcher: accetta .pdf o .png/.jpg."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return pdf_to_png_base64(path)
    return png_to_base64(path)


# ---------------------------------------------------------------------------
# Parsing della risposta LLM
# ---------------------------------------------------------------------------

def _estrai_blocco(testo: str, linguaggio: str) -> Optional[str]:
    """Estrae il primo blocco ```linguaggio ... ``` dalla risposta."""
    pattern = rf"```{linguaggio}\s*\n(.*?)```"
    match = re.search(pattern, testo, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def valida_sintassi_python(codice: str) -> Optional[str]:
    """
    Verifica che il codice sia Python sintatticamente valido senza eseguirlo.
    Restituisce None se OK, oppure il messaggio di errore.
    """
    try:
        ast.parse(codice)
        return None
    except SyntaxError as e:
        return f"SyntaxError riga {e.lineno}: {e.msg}"


# ---------------------------------------------------------------------------
# Funzione principale: genera codice CAD + parametri JSON da immagine
# ---------------------------------------------------------------------------

PROMPT_SISTEMA = (
    "Sei un esperto di disegno meccanico e modellazione CAD parametrica. "
    "Riceverai immagini di tavole tecniche meccaniche. "
    "Analizza la geometria del pezzo usando tutte le viste disponibili "
    "(frontale, laterale, sezione) e produci due output distinti come richiesto."
)

PROMPT_UTENTE = """Analizza questo disegno tecnico meccanico e produci ESATTAMENTE due output.

## OUTPUT 1 — Codice build123d (Python)

Scrivi una funzione `gen_step()` in build123d che ricrea la geometria del pezzo.

Regole:
- Usa SOLO funzioni build123d standard: Cylinder, Box, Extrude, Revolve, Hole, Fillet, Chamfer, Loft, Sweep, Shell, PolarLocations, GridLocations, BuildPart, BuildSketch, Plane, Circle, Rectangle, Polygon, add, subtract.
- Unità: millimetri.
- Se ci sono più viste (frontale, laterale, sezione), integrali per ricostruire il modello 3D.
- Se una dimensione non è leggibile, stima dalle proporzioni visive.
- Inizia sempre con `from build123d import *`.
- Il codice deve essere sintatticamente valido Python.

## OUTPUT 2 — Parametri strutturati (JSON)

Estrai un JSON con i parametri geometrici chiave:
{
    "part_type": "shaft|plate|housing|bracket|gear|flange|bushing|other",
    "overall_length_mm": <numero>,
    "max_diameter_mm": <numero o null se non cilindrico>,
    "max_width_mm": <numero o null se cilindrico>,
    "max_height_mm": <numero o null>,
    "features": ["bore", "thread", "flange", "step", "groove", "keyway", "fillet", "chamfer", "slot", "pocket", "rib", "hole"],
    "n_bores": <numero intero>,
    "n_steps": <numero di gradini/spalle>,
    "symmetry": "axial|planar|none",
    "is_hollow": true|false
}

Rispondi ESATTAMENTE in questo formato, senza testo aggiuntivo prima o dopo:

```python
<codice build123d qui>
```

```json
<JSON parametri qui>
```"""


def genera_cad_e_parametri(
    image_b64: str,
    model: str = "gpt-4.1",
    errore_precedente: Optional[str] = None,
) -> dict:
    """
    Genera codice build123d + JSON parametri da un'immagine base64 di disegno tecnico.

    Ritorna:
        {
            "code": str | None,          — codice build123d Python
            "params": dict | None,       — parametri strutturati JSON
            "syntax_error": str | None,  — errore di sintassi Python (se presente)
            "raw_response": str,         — risposta grezza del modello
        }
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY non impostata.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt_utente = PROMPT_UTENTE
    if errore_precedente:
        prompt_utente = (
            f"ATTENZIONE: il tentativo precedente ha prodotto questo errore Python:\n"
            f"{errore_precedente}\n\n"
            "Correggi il codice e riprova.\n\n"
        ) + prompt_utente

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=2500,
        messages=[
            {"role": "system", "content": PROMPT_SISTEMA},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt_utente},
                ],
            },
        ],
    )

    raw = (resp.choices[0].message.content or "").strip()

    codice = _estrai_blocco(raw, "python")
    params_raw = _estrai_blocco(raw, "json")

    params = None
    if params_raw:
        try:
            params = json.loads(params_raw)
        except json.JSONDecodeError as e:
            print(f"[parse] JSON non valido: {e}")

    syntax_error = valida_sintassi_python(codice) if codice else "blocco python non trovato nella risposta"

    return {
        "code": codice,
        "params": params,
        "syntax_error": syntax_error,
        "raw_response": raw,
    }


def genera_con_retry(
    image_b64: str,
    model: str = "gpt-4.1",
    max_tentativi: int = 3,
) -> dict:
    """
    Chiama genera_cad_e_parametri con retry automatico in caso di errore di sintassi.
    Al secondo tentativo passa l'errore al modello come feedback.
    """
    risultato = None
    for tentativo in range(1, max_tentativi + 1):
        errore_precedente = risultato["syntax_error"] if risultato and risultato.get("syntax_error") else None

        if tentativo > 1:
            print(f"  [retry {tentativo}/{max_tentativi}] errore precedente: {errore_precedente}")

        risultato = genera_cad_e_parametri(image_b64, model=model, errore_precedente=errore_precedente)

        if not risultato["syntax_error"]:
            if tentativo > 1:
                print(f"  [retry] successo al tentativo {tentativo}")
            return risultato

    print(f"  [retry] fallito dopo {max_tentativi} tentativi")
    return risultato


# ---------------------------------------------------------------------------
# CLI: test su un singolo file
# ---------------------------------------------------------------------------

def _stampa_risultato(path: str, risultato: dict):
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"File: {path}")
    print(sep)

    if risultato["syntax_error"]:
        print(f"  [ERRORE] Sintassi: {risultato['syntax_error']}")
    else:
        print("  [OK] Codice Python valido")

    if risultato["params"]:
        p = risultato["params"]
        print(f"  part_type      : {p.get('part_type')}")
        print(f"  overall_length : {p.get('overall_length_mm')} mm")
        print(f"  max_diameter   : {p.get('max_diameter_mm')} mm")
        print(f"  features       : {p.get('features')}")
        print(f"  n_bores        : {p.get('n_bores')}")
        print(f"  symmetry       : {p.get('symmetry')}")
    else:
        print("  [WARN] JSON parametri non estratto")

    if risultato["code"]:
        print(f"\n--- Codice build123d ({len(risultato['code'].splitlines())} righe) ---")
        print(risultato["code"].encode("ascii", errors="replace").decode("ascii"))
    else:
        print("\n--- Risposta grezza del modello ---")
        print(risultato["raw_response"].encode("ascii", errors="replace").decode("ascii"))
    print(sep)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python poc_phase0.py <file.pdf | file.png>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Carico immagine da: {path}")

    image_b64 = image_path_to_base64(path)
    if not image_b64:
        print("Errore: impossibile caricare l'immagine.")
        sys.exit(1)

    print("Chiamo GPT-4.1 per generare codice CAD + parametri...")
    risultato = genera_con_retry(image_b64)
    _stampa_risultato(path, risultato)
