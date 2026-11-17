"""
Classifica i disegni dell'archivio locale con un decision model Liquid
(attributi strutturati: tipo di pezzo, classe di forma, diametri, proporzioni,
dettagli) e misura se questi attributi aiuterebbero la ricerca.

Tutto in locale: i PDF vengono letti dalle cartelle del progetto, le immagini
e le risposte restano in eval/candidates/ (ignorato da git). Weaviate non
viene toccato.

Uso:
    python eval/classify_archive.py classify --only-tests        # solo disegni dei test case
    python eval/classify_archive.py classify                     # tutto l'archivio locale
    python eval/classify_archive.py report                       # analisi
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import run_eval as r  # noqa: E402
from judge_eval import load_dotenv  # noqa: E402

PROJECT = EVAL_DIR.parent.parent  # C:\Progetti_IA\Sinde
ARCHIVE_DIRS = [
    PROJECT / "wetransfer_pdf_2025-11-06_1344",
    PROJECT / "Migliorie aprile 2026" / "disegni",
]
WORK = EVAL_DIR / "candidates" / "_archive"
IMG_DIR = WORK / "img"
ATTR_DIR = WORK / "attrs"
CAND_DIR = EVAL_DIR / "candidates"

STATE_TEXT = (
    "Disegno tecnico di un componente meccanico. Valuta solo la geometria del pezzo: "
    "ignora cartiglio, tabelle, testi e note."
)

QUESTIONS: Dict[str, Dict[str, Any]] = {
    "part_type": {
        "type": "choice",
        "instructions": "Che tipo di componente rappresenta il disegno?",
        "criteria": {
            "pignone_ingranaggio": "ruota dentata, pignone, ingranaggio, corona",
            "albero": "albero o asse lungo con gradini, sedi cuscinetto, cave",
            "perno": "perno o spina corta",
            "canotto_boccola_distanziale": "canotto, boccola, bussola, distanziale, anello tubolare",
            "nipplo_raccordo": "nipplo, raccordo, attacco filettato per fluidi",
            "carrucola_puleggia": "carrucola, puleggia, rullo con gola",
            "flangia_disco": "flangia, disco, coperchio circolare piatto",
            "staffa_supporto": "staffa, supporto, mensola, forcella",
            "blocchetto": "blocchetto o corpo prismatico massiccio fresato",
            "piastra_lamiera": "piastra, lamiera piegata o tagliata",
            "altro": "nessuna delle precedenti",
        },
    },
    "shape_class": {
        "type": "choice",
        "instructions": "Qual è la classe di forma del pezzo?",
        "criteria": {
            "tornito": "pezzo tornito/assialsimmetrico, sezioni circolari, diametri (simbolo Ø)",
            "prismatico": "pezzo prismatico/fresato dal pieno",
            "lamiera": "lamiera tagliata o piegata, spessore costante",
            "misto": "combinazione evidente di parti tornite e prismatiche",
        },
    },
    "n_diameters": {
        "type": "choice",
        "instructions": "Quanti diametri o gradini diversi ha il pezzo?",
        "criteria": {
            "nessuno": "nessun diametro (pezzo non tornito)",
            "1-2": "uno o due diametri",
            "3-5": "da tre a cinque diametri o gradini",
            "oltre_5": "più di cinque diametri o gradini",
        },
    },
    "proportions": {
        "type": "choice",
        "instructions": "Che proporzioni ha il pezzo?",
        "criteria": {
            "allungato": "lunghezza molto maggiore della sezione (es. alberi, perni lunghi)",
            "bilanciato": "dimensioni principali confrontabili",
            "piatto": "spessore molto minore delle altre dimensioni (dischi, piastre)",
        },
    },
    "has_teeth": {"type": "noul", "instructions": "Il pezzo ha una dentatura (denti di ingranaggio o scanalato)?"},
    "has_keyway": {"type": "noul", "instructions": "Il pezzo ha una cava per chiavetta o linguetta?"},
    "has_thread": {"type": "noul", "instructions": "Il pezzo ha filettature?"},
    "has_through_holes": {"type": "noul", "instructions": "Il pezzo ha fori passanti di fissaggio?"},
}


def list_archive() -> List[Path]:
    files: List[Path] = []
    for d in ARCHIVE_DIRS:
        files += [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(files)


def archive_key(name: str) -> str:
    """Codice del pezzo; per i nomi fuori standard (es. BUSSO00043) usa il nome file senza estensione."""
    return r.code_key(name) or Path(name).stem.upper()


def file_id(p: Path) -> str:
    rel = p.relative_to(PROJECT).as_posix()
    return rel.replace("/", "__")


_RENDER_LOCK = threading.Lock()


def render(pdf: Path, out: Path, max_side: int = 1600) -> bool:
    # pypdfium2 non è thread-safe: rendering serializzato, le chiamate API restano parallele.
    with _RENDER_LOCK:
        return _render(pdf, out, max_side)


def _render(pdf: Path, out: Path, max_side: int) -> bool:
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(str(pdf))
        page = doc[0]
        w, h = page.get_size()
        scale = max_side / max(w, h)
        img = page.render(scale=scale).to_pil().convert("L")
        img.thumbnail((max_side, max_side))
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, format="JPEG", quality=88)
        return True
    except Exception as exc:
        print(f"   render fallito {pdf.name}: {exc}")
        return False


def classify(img: Path, model: str) -> Dict[str, Any]:
    key = os.environ.get("LIQUID_API_KEY")
    if not key:
        raise RuntimeError("LIQUID_API_KEY mancante (eval/.env)")
    body = {
        "model": model,
        "state": STATE_TEXT,
        "images": ["data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode("ascii")],
        "questions": QUESTIONS,
    }
    for attempt in range(6):
        resp = requests.post(
            "https://api.liquid.ai/decisions/v1/systemone",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        out: Dict[str, Any] = {"_input_tokens": data.get("usage", {}).get("input_tokens")}
        for name, a in data.get("answers", {}).items():
            if a.get("type") == "choice":
                out[name] = {"choice": a.get("choice"), "confidence": a.get("confidence"), "probabilities": a.get("probabilities")}
            elif a.get("type") == "noul":
                out[name] = a.get("noul")
        return out
    raise RuntimeError("troppi tentativi (429/5xx)")


def test_case_keys() -> set:
    keys = set()
    for tc in json.loads(r.DEFAULT_TEST_CASES.read_text(encoding="utf-8")):
        for c in [tc["input"], *tc["expected"], *tc["unwanted"]]:
            keys.add(r.code_key(c))
    return keys


def cmd_classify(args) -> int:
    files = list_archive()
    if args.only_tests:
        wanted = test_case_keys()
        files = [f for f in files if r.code_key(f.name) in wanted]
    print(f"PDF da classificare: {len(files)}")
    todo = []
    for f in files:
        fid = file_id(f)
        if not (ATTR_DIR / f"{fid}.json").exists():
            todo.append((f, fid))
    print(f"   già in cache: {len(files) - len(todo)}, da fare: {len(todo)}")

    def work(item):
        f, fid = item
        img = IMG_DIR / f"{fid}.jpg"
        if not img.exists() and not render(f, img):
            return fid, None, "render"
        res = classify(img, args.model)
        res["_file"] = f.relative_to(PROJECT).as_posix()
        res["_key"] = archive_key(f.name)
        res["_name"] = f.name
        ATTR_DIR.mkdir(parents=True, exist_ok=True)
        (ATTR_DIR / f"{fid}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return fid, res, None

    done = errors = 0
    tokens = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for fu in as_completed(futs):
            try:
                fid, res, err = fu.result()
                if err:
                    errors += 1
                else:
                    done += 1
                    tokens += res.get("_input_tokens") or 0
            except Exception as exc:
                errors += 1
                print(f"   errore: {exc}")
            if (done + errors) % 100 == 0:
                print(f"   {done + errors}/{len(todo)}  (token {tokens}, costo ~${tokens / 1e6 * 0.042:.3f})", flush=True)
    print(f"Fatto: {done} classificati, {errors} errori, token {tokens}, costo ~${tokens / 1e6 * 0.042:.3f}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def load_attrs() -> Dict[str, List[Dict[str, Any]]]:
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for f in ATTR_DIR.glob("*.json"):
        a = json.loads(f.read_text(encoding="utf-8"))
        a["_key"] = archive_key(a.get("_name") or f.stem)
        by_key.setdefault(a["_key"], []).append(a)
    return by_key


def ch(a: Optional[Dict[str, Any]], q: str) -> Optional[str]:
    return (a or {}).get(q, {}).get("choice") if isinstance((a or {}).get(q), dict) else None


def compat(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "type": ch(a, "part_type") == ch(b, "part_type"),
        "shape": ch(a, "shape_class") == ch(b, "shape_class"),
        "diam": ch(a, "n_diameters") == ch(b, "n_diameters"),
        "prop": ch(a, "proportions") == ch(b, "proportions"),
    }


def fmt(a: Optional[Dict[str, Any]]) -> str:
    if not a:
        return "(non classificato)"
    return f"{ch(a, 'part_type')}/{ch(a, 'shape_class')}/{ch(a, 'n_diameters')}/{ch(a, 'proportions')}"


def cmd_report(args) -> int:
    attrs = load_attrs()
    all_items = [v[0] for v in attrs.values()]
    print(f"Disegni classificati (codici distinti): {len(all_items)}")
    print("Distribuzione tipo di pezzo:", Counter(ch(a, "part_type") for a in all_items).most_common())
    print("Distribuzione forma:", Counter(ch(a, "shape_class") for a in all_items).most_common())

    tcs = json.loads(r.DEFAULT_TEST_CASES.read_text(encoding="utf-8"))
    tot_exp = {"type": 0, "shape": 0, "type+shape": 0, "n": 0}
    tot_unw = {"type": 0, "shape": 0, "type+shape": 0, "n": 0}
    for tc in tcs:
        q = (attrs.get(r.code_key(tc["input"])) or [None])[0]
        print(f"\n[{tc['input']}] input: {fmt(q)}")
        if not q:
            continue
        same_class = [a for a in all_items if a["_key"] != q["_key"] and ch(a, "part_type") == ch(q, "part_type") and ch(a, "shape_class") == ch(q, "shape_class")]
        print(f"   archivio con stesso tipo+forma: {len(same_class)} su {len(all_items)} ({len(same_class) / max(len(all_items), 1):.0%})")
        # pool congelato top-50: quanti candidati sarebbero filtrati
        mf = CAND_DIR / tc["input"] / "manifest.json"
        if mf.exists():
            cands = [c for c in json.loads(mf.read_text(encoding="utf-8"))["candidates"] if c["key"] != q["_key"]]
            cls = [(attrs.get(c["key"]) or [None])[0] for c in cands]
            known = [x for x in cls if x]
            kept = [x for x in known if ch(x, "part_type") == ch(q, "part_type") and ch(x, "shape_class") == ch(q, "shape_class")]
            print(f"   pool top-50 attuale: {len(known)} classificati, {len(kept)} con stesso tipo+forma")
        for label, codes, tot in (("atteso", tc["expected"], tot_exp), ("indesiderato", tc["unwanted"], tot_unw)):
            for c in codes:
                a = (attrs.get(r.code_key(c)) or [None])[0]
                if not a:
                    print(f"   {label:>12} {c}: non classificato")
                    continue
                m = compat(q, a)
                tot["n"] += 1
                tot["type"] += m["type"]
                tot["shape"] += m["shape"]
                tot["type+shape"] += m["type"] and m["shape"]
                print(f"   {label:>12} {c}: {fmt(a)}  stesso tipo={m['type']} forma={m['shape']} diametri={m['diam']} proporzioni={m['prop']}")

    def pct(t, k):
        return f"{t[k]}/{t['n']}" if t["n"] else "-"

    print("\n=== Riepilogo compatibilità con il disegno di input ===")
    print(f"   attesi       stesso tipo {pct(tot_exp, 'type')}, stessa forma {pct(tot_exp, 'shape')}, entrambi {pct(tot_exp, 'type+shape')}")
    print(f"   indesiderati stesso tipo {pct(tot_unw, 'type')}, stessa forma {pct(tot_unw, 'shape')}, entrambi {pct(tot_unw, 'type+shape')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--only-tests", action="store_true")
    c.add_argument("--model", default="d1.2-ultra")
    c.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    args = ap.parse_args()
    load_dotenv(EVAL_DIR / ".env")
    return cmd_classify(args) if args.cmd == "classify" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
