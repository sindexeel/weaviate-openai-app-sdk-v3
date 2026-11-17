"""
Copre di bianco il cartiglio (logo, descrizione, codici, materiale, tolleranze,
firme) nelle immagini dei candidati congelati, così i giudici vedono solo la
geometria del pezzo, come nel caso reale di un disegno nuovo senza cartiglio.

Il riquadro del cartiglio è individuato da Liquid (domanda di tipo "box").
Output: eval/candidates/<input>/masked/NN.jpg e un foglio di controllo
eval/candidates/_mask_check.jpg con un campione di immagini mascherate.

Uso:
    python eval/mask_titleblock.py
    python eval/mask_titleblock.py --sheet-only
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PIL import Image, ImageDraw

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
from judge_eval import load_dotenv  # noqa: E402

CAND_DIR = EVAL_DIR / "candidates"

BOX_INSTRUCTIONS = (
    "Riquadro che contiene il cartiglio del disegno: tabelle con logo dell'azienda, descrizione, "
    "numero di disegno, codice, materiale, tolleranze generali, data, firme e note legali. "
    "Non includere le viste del pezzo."
)
MARGIN = 0.015      # allargamento del riquadro (frazione dell'immagine)
EDGE_SNAP = 0.08    # se il riquadro è vicino al bordo, lo estende fino al bordo


def detect_box(img: Path, model: str) -> Dict:
    body = {
        "model": model,
        "state": "Disegno tecnico meccanico.",
        "images": ["data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode("ascii")],
        "questions": {"cartiglio": {"type": "box", "instructions": BOX_INSTRUCTIONS}},
    }
    for attempt in range(6):
        resp = requests.post(
            "https://api.liquid.ai/decisions/v1/systemone",
            headers={"Authorization": f"Bearer {os.environ['LIQUID_API_KEY']}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json()["answers"]["cartiglio"]
    raise RuntimeError("troppi tentativi")


def apply_mask(src: Path, dst: Path, ans: Dict) -> Optional[tuple]:
    img = Image.open(src).convert("L")
    w, h = img.size
    if not ans.get("box") or (ans.get("found") or 0) < 0.5:
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, format="JPEG", quality=88)
        return None
    b = ans["box"]
    top, bottom = max(0.0, b["top"] - MARGIN), min(1.0, b["bottom"] + MARGIN)
    left, right = max(0.0, b["left"] - MARGIN), min(1.0, b["right"] + MARGIN)
    if 1.0 - bottom < EDGE_SNAP:
        bottom = 1.0
    if right > 1.0 - EDGE_SNAP:
        right = 1.0
    if left < EDGE_SNAP:
        left = 0.0
    if top < EDGE_SNAP:
        top = 0.0
    box = (int(left * w), int(top * h), int(right * w), int(bottom * h))
    ImageDraw.Draw(img).rectangle(box, fill=255)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="JPEG", quality=88)
    return box


def make_sheet(paths: List[Path], out: Path, n: int = 16, seed: int = 7) -> None:
    random.Random(seed).shuffle(paths)
    sample = paths[:n]
    thumbs = []
    for p in sample:
        im = Image.open(p).convert("L")
        im.thumbnail((360, 360))
        canvas = Image.new("L", (370, 370), 160)
        canvas.paste(im, (5, 5))
        thumbs.append(canvas)
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("L", (cols * 370, rows * 370), 160)
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 370, (i // cols) * 370))
    sheet.save(out, format="JPEG", quality=85)
    print(f"Foglio di controllo: {out} ({len(thumbs)} immagini)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="d1.2-ultra")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sheet-only", action="store_true")
    args = ap.parse_args()
    load_dotenv(EVAL_DIR / ".env")

    srcs = sorted(p for p in CAND_DIR.glob("*/*.jpg") if not p.parent.name.startswith("_"))
    if not args.sheet_only:
        todo = []
        for s in srcs:
            box_file = s.parent / "masked" / f"{s.stem}.box.json"
            if not (s.parent / "masked" / s.name).exists():
                todo.append((s, box_file))
        print(f"Immagini: {len(srcs)}, da mascherare: {len(todo)}")

        def work(item):
            s, box_file = item
            if box_file.exists():
                ans = json.loads(box_file.read_text(encoding="utf-8"))
            else:
                ans = detect_box(s, args.model)
                box_file.parent.mkdir(parents=True, exist_ok=True)
                box_file.write_text(json.dumps(ans), encoding="utf-8")
            return apply_mask(s, s.parent / "masked" / s.name, ans)

        not_found = errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fu in as_completed([ex.submit(work, it) for it in todo]):
                try:
                    if fu.result() is None:
                        not_found += 1
                except Exception as exc:
                    errors += 1
                    print(f"   errore: {exc}")
        print(f"Fatto. Cartiglio non trovato: {not_found}, errori: {errors}")

    masked = sorted(CAND_DIR.glob("*/masked/*.jpg"))
    make_sheet(masked, CAND_DIR / "_mask_check.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
