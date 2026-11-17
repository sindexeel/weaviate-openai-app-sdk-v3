"""
Test realistico end-to-end: il disegno di input NON ha il cartiglio (come un
DWG 2D di un pezzo nuovo convertito in immagine), l'archivio invece sì.

Per ogni test case:
  1. input = immagine del disegno con il cartiglio coperto (mask_titleblock.py)
  2. RICERCA   : l'input pulito viene caricato sul sistema attuale
                 (/upload-image + /image-search, solo lettura) -> pool top-N
  3. RICONOSCIMENTO: Liquid classifica l'input pulito (tipo, forma, ...)
  4. GIUDICE   : Liquid confronta l'input pulito con ogni candidato
                 (A) pool della ricerca, (B) tutti i disegni dell'archivio
                 locale dello stesso tipo, (C) unione di A e B
  5. MISURA    : stesse metriche e ground truth di run_eval.py

Il disegno di input esiste anche in archivio: viene escluso dalle metriche
(nella realtà il pezzo nuovo non c'è).

Uso:
    python eval/realistic_eval.py
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import run_eval as r  # noqa: E402
import classify_archive as ca  # noqa: E402
from judge_eval import judge_liquid, load_dotenv, rank_metrics, mean  # noqa: E402

CAND_DIR = EVAL_DIR / "candidates"
REAL_DIR = CAND_DIR / "_real"
RESULTS_DIR = EVAL_DIR / "results"


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def search_pool(session, base_url: str, query_img: Path, limit: int, out_dir: Path) -> List[Dict[str, Any]]:
    manifest = out_dir / "pool.json"
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    with query_img.open("rb") as fh:
        up = session.post(f"{base_url}/upload-image", files={"image": (query_img.name, fh, "image/jpeg")}, timeout=600)
    up.raise_for_status()
    image_id = up.json()["image_id"]
    sr = session.post(f"{base_url}/image-search", json={"collection": "Sinde5", "image_id": image_id, "limit": limit}, timeout=600)
    sr.raise_for_status()
    results = sr.json().get("results") or []
    (out_dir / "pool").mkdir(parents=True, exist_ok=True)
    items = []
    for i, res in enumerate(results):
        props = res.get("properties") or {}
        name = r.result_name(res)
        img_path = None
        if props.get("image_b64"):
            try:
                im = Image.open(BytesIO(base64.b64decode(props["image_b64"]))).convert("L")
                im.thumbnail((1600, 1600))
                img_path = out_dir / "pool" / f"{i:02d}.jpg"
                im.save(img_path, format="JPEG", quality=88)
            except Exception as exc:
                print(f"   immagine non leggibile {name}: {exc}")
        items.append({"rank": i + 1, "name": name, "key": ca.archive_key(name), "image": str(img_path) if img_path else None})
    manifest.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    return items


def class_candidates(qtype: str, attrs: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for key, lst in attrs.items():
        a = lst[0]
        if ca.ch(a, "part_type") != qtype:
            continue
        img = ca.IMG_DIR / f"{ca.file_id(ca.PROJECT / a['_file'])}.jpg"
        if img.exists():
            out.append({"name": a["_name"], "key": key, "image": str(img)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=r.DEFAULT_BASE_URL)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--model", default="d1.2-ultra")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_dotenv(EVAL_DIR / ".env")

    tcs = json.loads(r.DEFAULT_TEST_CASES.read_text(encoding="utf-8"))
    attrs = ca.load_attrs()
    session = requests.Session()
    base = args.base_url.rstrip("/")
    report: Dict[str, Any] = {"cases": []}

    for tc in tcs:
        inp = tc["input"]
        mf = json.loads((CAND_DIR / inp / "manifest.json").read_text(encoding="utf-8"))
        query_img = CAND_DIR / inp / "masked" / mf["query_image"]
        out_dir = REAL_DIR / inp
        out_dir.mkdir(parents=True, exist_ok=True)
        input_key = ca.archive_key(inp)
        print(f"\n[{inp}]", flush=True)

        # 2. ricerca partendo dal disegno pulito
        pool = search_pool(session, base, query_img, args.limit, out_dir)
        pool_c = [p for p in pool if p["key"] != input_key and p["image"]]
        self_rank = next((p["rank"] for p in pool if p["key"] == input_key), None)
        print(f"   ricerca: {len(pool)} risultati, disegno stesso in posizione {self_rank or '-'}")

        # 3. riconoscimento del disegno pulito
        cls_file = out_dir / "query_class.json"
        if cls_file.exists():
            qcls = json.loads(cls_file.read_text(encoding="utf-8"))
        else:
            qcls = ca.classify(query_img, args.model)
            cls_file.write_text(json.dumps(qcls, ensure_ascii=False), encoding="utf-8")
        qtype = ca.ch(qcls, "part_type")
        print(f"   riconosciuto come: {ca.fmt(qcls)} (conf. tipo {qcls['part_type']['confidence']:.2f})")

        # 4. giudice su pool + classe
        klass = [c for c in class_candidates(qtype, attrs) if c["key"] != input_key]
        print(f"   classe '{qtype}' nell'archivio locale: {len(klass)} disegni")
        judge_dir = out_dir / "judge"
        judge_dir.mkdir(exist_ok=True)
        todo = {}
        for c in pool_c + klass:
            f = judge_dir / f"{safe(c['key'])}.json"
            if not f.exists() and c["key"] not in todo:
                todo[c["key"]] = (Path(c["image"]), f)
        print(f"   coppie da giudicare: {len(todo)}", flush=True)
        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_liquid, query_img, img, args.model): f for img, f in todo.values()}
            for fu in as_completed(futs):
                try:
                    futs[fu].write_text(json.dumps(fu.result()), encoding="utf-8")
                except Exception as exc:
                    errors += 1
                    print(f"   errore giudice: {exc}")
        if errors:
            print(f"   {errors} coppie in errore (rilanciare per completare)")

        def p_of(key: str) -> float:
            f = judge_dir / f"{safe(key)}.json"
            return float(json.loads(f.read_text(encoding="utf-8")).get("similar", 0.0)) if f.exists() else -1.0

        def dedupe(keys: List[str]) -> List[str]:
            seen, out = set(), []
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
            return out

        variants = {
            "ricerca_attuale": dedupe([p["key"] for p in pool_c]),
            "A_pool+giudice": sorted(dedupe([p["key"] for p in pool_c]), key=p_of, reverse=True),
            "B_classe+giudice": sorted(dedupe([c["key"] for c in klass]), key=p_of, reverse=True),
            "C_unione+giudice": sorted(dedupe([p["key"] for p in pool_c] + [c["key"] for c in klass]), key=p_of, reverse=True),
        }
        # le chiavi dell'archivio locale per i codici standard coincidono con code_key
        exp_in = {
            "pool": [e for e in tc["expected"] if r.code_key(e) in variants["ricerca_attuale"]],
            "classe": [e for e in tc["expected"] if r.code_key(e) in variants["B_classe+giudice"]],
        }
        row = {"input": inp, "self_rank_search": self_rank, "query_class": ca.fmt(qcls), "class_size": len(klass),
               "expected_in_pool": exp_in["pool"], "expected_in_class": exp_in["classe"], "variants": {}}
        for vname, keys in variants.items():
            m = rank_metrics(tc, keys)
            row["variants"][vname] = m
            print(f"   {vname:>18}: attesi[{r.fmt_ranks(m['expected_ranks'])}]  indesiderati[{r.fmt_ranks(m['unwanted_ranks'])}]")
        report["cases"].append(row)

    print("\n=== Riepilogo (input senza cartiglio) ===")
    summary = {}
    for vname in report["cases"][0]["variants"]:
        rows = [c["variants"][vname] for c in report["cases"] if c["variants"][vname]["recall@10"] is not None]
        summary[vname] = {
            "recall@5": mean([x["recall@5"] for x in rows]),
            "recall@10": mean([x["recall@10"] for x in rows]),
            "recall@20": mean([x["recall@20"] for x in rows]),
            "mrr": mean([x["mrr"] for x in rows]),
            "unwanted@10": sum(c["variants"][vname]["unwanted@10"] for c in report["cases"]),
        }
        print(f"{vname:>18}: " + "  ".join(f"{k}={v}" for k, v in summary[vname].items()))
    report["summary"] = summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"realistic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRisultati salvati in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
