"""
Congela i candidati della ricerca attuale per l'esperimento con i giudici.

Per ogni test case carica il PDF di input su /upload-image, chiama
/image-search con un limite alto (default 50) e salva in eval/candidates/:
  - <input>/manifest.json : ranking originale (nome, punteggio)
  - <input>/NN.jpg        : immagini ridimensionate dei candidati
Il primo risultato (il disegno stesso, sempre in testa) è l'immagine di query.

Congelare i candidati è necessario: la caption GPT varia tra run, quindi
senza un set fisso i giudici non vedrebbero gli stessi candidati.

Uso:
    python eval/freeze_candidates.py --limit 50
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval as r  # noqa: E402

CAND_DIR = Path(__file__).resolve().parent / "candidates"


def save_image(b64: str, path: Path, max_side: int) -> bool:
    try:
        img = Image.open(BytesIO(base64.b64decode(b64))).convert("L")
    except Exception as exc:
        print(f"   immagine non leggibile ({path.name}): {exc}")
        return False
    img.thumbnail((max_side, max_side))
    img.save(path, format="JPEG", quality=88)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=r.DEFAULT_BASE_URL)
    ap.add_argument("--collection", default="Sinde5")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-side", type=int, default=1600, help="lato massimo delle immagini salvate (px)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    cases = json.loads(r.DEFAULT_TEST_CASES.read_text(encoding="utf-8"))
    session = requests.Session()
    base = args.base_url.rstrip("/")
    for tc in cases:
        pdf = r.find_input_pdf(r.DEFAULT_PDF_DIR, tc["input"])
        if not pdf:
            print(f"[{tc['input']}] PDF non trovato")
            continue
        print(f"[{tc['input']}] ricerca top-{args.limit} ...", flush=True)
        out = r.run_case(session, base, args.collection, args.limit, pdf, args.timeout)
        case_dir = CAND_DIR / tc["input"]
        case_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for i, res in enumerate(out["results"]):
            props = res.get("properties") or {}
            name = r.result_name(res)
            img_file = f"{i:02d}.jpg"
            ok = bool(props.get("image_b64")) and save_image(props["image_b64"], case_dir / img_file, args.max_side)
            items.append({
                "rank": i + 1,
                "name": name,
                "key": r.code_key(name),
                "score": res.get("bm25_score"),
                "image": img_file if ok else None,
            })
        input_key = r.code_key(tc["input"])
        query = next((it for it in items if it["key"] == input_key and it["image"]), None)
        manifest = {
            "input": tc["input"],
            "mode": out["mode"],
            "limit": args.limit,
            "query_image": query["image"] if query else None,
            "candidates": items,
        }
        (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        n_img = sum(1 for it in items if it["image"])
        exp_in_pool = [c for c in tc["expected"] if any(it["key"] == r.code_key(c) for it in items)]
        print(f"   [{out['mode']}] {len(items)} candidati, {n_img} immagini, query={'ok' if query else 'MANCANTE'}, "
              f"attesi nel pool: {len(exp_in_pool)}/{len(tc['expected'])} {exp_in_pool}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
