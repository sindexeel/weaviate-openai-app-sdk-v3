"""
Runner di valutazione della ricerca Sinde sui test case del cliente.

Replica il flusso del widget: per ogni test case carica il PDF di input su
/upload-image, chiama /image-search e confronta i risultati con i pezzi
"molto simili" (expected) e "non simili" (unwanted) di test_cases.json.

Uso:
    python eval/run_eval.py
    python eval/run_eval.py --base-url http://localhost:10000 --collection Sinde2 --limit 20
    python eval/run_eval.py --pdf-dir "../wetransfer_pdf_2025-11-06_1344" --label baseline

Il confronto dei codici ignora la revisione (es. PF00017176-A == PF00017176,
702.0144.00 == 702.0144.00.0), perché nel test del cliente le revisioni sono
scritte in modo non uniforme.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_CASES = ROOT / "weaviate-image-app" / "public" / "test_cases.json"
DEFAULT_PDF_DIR = ROOT.parent / "wetransfer_pdf_2025-11-06_1344"
DEFAULT_BASE_URL = "https://weaviate-openai-app-sdk-v3.onrender.com"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

_DOT_CODE = re.compile(r"(\d{3}\.\d{4}\.\d{2})")
_ALPHA_CODE = re.compile(r"([A-Z]{1,2}\d{8})")


def code_key(text: str) -> Optional[str]:
    """Chiave del pezzo senza revisione/estensione: '702.0144.00' o 'PF00017176'."""
    if not text:
        return None
    t = Path(str(text)).name.upper()
    m = _DOT_CODE.search(t)
    if m:
        return m.group(1)
    m = _ALPHA_CODE.search(t)
    if m:
        return m.group(1)
    return None


def find_input_pdf(pdf_dir: Path, code: str) -> Optional[Path]:
    key = code_key(code)
    candidates = sorted(
        p for p in pdf_dir.rglob("*") if p.suffix.lower() == ".pdf" and code_key(p.name) == key
    )
    return candidates[0] if candidates else None


def result_name(r: Dict[str, Any]) -> str:
    props = r.get("properties") or {}
    return str(props.get("name") or props.get("source_pdf") or "")


def run_case(
    session: requests.Session,
    base_url: str,
    collection: str,
    limit: int,
    pdf_path: Path,
    timeout: int,
) -> Dict[str, Any]:
    with pdf_path.open("rb") as fh:
        up = session.post(
            f"{base_url}/upload-image",
            files={"image": (pdf_path.name, fh, "application/pdf")},
            timeout=timeout,
        )
    up.raise_for_status()
    image_id = up.json().get("image_id")
    if not image_id:
        raise RuntimeError(f"upload senza image_id: {up.text[:200]}")

    t0 = time.time()
    sr = session.post(
        f"{base_url}/image-search",
        json={"collection": collection, "image_id": image_id, "limit": limit},
        timeout=timeout,
    )
    elapsed = time.time() - t0
    sr.raise_for_status()
    data = sr.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    results = data.get("results") or []
    # La fusion DINOv2+caption restituisce uuid vuoto e distance None; il fallback caption-only no.
    if results:
        mode = "fusion" if all(not r.get("uuid") and r.get("distance") is None for r in results) else "caption-hybrid"
    else:
        mode = "empty"
    trimmed = [
        {
            "name": result_name(r),
            "uuid": r.get("uuid"),
            "distance": r.get("distance"),
            "bm25_score": r.get("bm25_score"),
            "dim_similarity": r.get("dim_similarity"),
            "combined_score": r.get("combined_score"),
        }
        for r in results
    ]
    return {"results": results, "raw": trimmed, "mode": mode, "search_seconds": round(elapsed, 1)}


def score_case(tc: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_key = code_key(tc["input"])
    names = [result_name(r) for r in results]
    keys = [code_key(n) for n in names]

    # Il disegno di input è in archivio: lo togliamo dal ranking per non falsare le metriche.
    ranked = [(n, k) for n, k in zip(names, keys) if k != input_key]
    self_rank = next((i + 1 for i, k in enumerate(keys) if k == input_key), None)

    def rank_of(code: str) -> Optional[int]:
        ck = code_key(code)
        return next((i + 1 for i, (_, k) in enumerate(ranked) if k == ck), None)

    expected = {c: rank_of(c) for c in tc.get("expected", [])}
    unwanted = {c: rank_of(c) for c in tc.get("unwanted", [])}

    def recall_at(k: int) -> Optional[float]:
        if not expected:
            return None
        return sum(1 for r in expected.values() if r and r <= k) / len(expected)

    def unwanted_at(k: int) -> int:
        return sum(1 for r in unwanted.values() if r and r <= k)

    found = [r for r in expected.values() if r]
    return {
        "input": tc["input"],
        "self_rank": self_rank,
        "expected_ranks": expected,
        "unwanted_ranks": unwanted,
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "recall@20": recall_at(20),
        "unwanted@5": unwanted_at(5),
        "unwanted@10": unwanted_at(10),
        "mrr": (1.0 / min(found)) if found else 0.0,
        "top10": [n for n, _ in ranked[:10]],
    }


def fmt_ranks(d: Dict[str, Optional[int]]) -> str:
    return ", ".join(f"{c}:{r if r else '-'}" for c, r in d.items()) or "(nessuno)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument(
        "--collection",
        default="Sinde5",
        help="collection caption (default Render = Sinde5; con FUSION_ENABLED il server usa comunque la sua default)",
    )
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--test-cases", type=Path, default=DEFAULT_TEST_CASES)
    ap.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--label", default="run", help="etichetta del file risultati")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    cases = json.loads(args.test_cases.read_text(encoding="utf-8"))
    session = requests.Session()

    # Risveglia il servizio Render (piano free: cold start lento).
    try:
        h = session.get(f"{base_url}/health", timeout=args.timeout)
        print(f"[health] {h.status_code} {h.text[:120]}")
    except Exception as exc:
        print(f"[health] errore: {exc}")
        return 1

    scored: List[Dict[str, Any]] = []
    for tc in cases:
        pdf = find_input_pdf(args.pdf_dir, tc["input"])
        if not pdf:
            print(f"[{tc['input']}] PDF di input non trovato in {args.pdf_dir}")
            scored.append({"input": tc["input"], "error": "input pdf not found"})
            continue
        print(f"[{tc['input']}] ricerca con {pdf.name} ...", flush=True)
        try:
            out = run_case(session, base_url, args.collection, args.limit, pdf, args.timeout)
        except Exception as exc:
            print(f"   errore: {exc}")
            scored.append({"input": tc["input"], "error": str(exc)})
            continue
        s = score_case(tc, out["results"])
        s["search_seconds"] = out["search_seconds"]
        s["n_results"] = len(out["results"])
        s["mode"] = out["mode"]
        s["raw"] = out["raw"]
        scored.append(s)
        print(
            f"   [{s['mode']}] self:{s['self_rank'] or '-'}  attesi[{fmt_ranks(s['expected_ranks'])}]  "
            f"indesiderati[{fmt_ranks(s['unwanted_ranks'])}]  ({s['search_seconds']}s)"
        )

    ok = [s for s in scored if "error" not in s]
    with_exp = [s for s in ok if s["recall@10"] is not None]

    def avg(key: str, rows: List[Dict[str, Any]]) -> Optional[float]:
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else None

    summary = {
        "cases_ok": len(ok),
        "cases_total": len(scored),
        "recall@5": avg("recall@5", with_exp),
        "recall@10": avg("recall@10", with_exp),
        "recall@20": avg("recall@20", with_exp),
        "mrr": avg("mrr", with_exp),
        "unwanted@5_total": sum(s["unwanted@5"] for s in ok),
        "unwanted@10_total": sum(s["unwanted@10"] for s in ok),
        "unwanted_total": sum(len(s["unwanted_ranks"]) for s in ok),
        "expected_total": sum(len(s["expected_ranks"]) for s in ok),
    }

    print("\n=== Riepilogo ===")
    for k, v in summary.items():
        print(f"{k:>18}: {v}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{args.label}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "base_url": base_url,
                    "collection": args.collection,
                    "limit": args.limit,
                    "test_cases": str(args.test_cases),
                    "timestamp": stamp,
                },
                "summary": summary,
                "cases": scored,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nRisultati salvati in {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
