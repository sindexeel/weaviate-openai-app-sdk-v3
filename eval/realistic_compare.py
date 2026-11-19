"""
Confronto realistico (input senza cartiglio) tra tre strategie:
  1. ricerca top-50 + giudice
  2. ricerca allargata (fino a 100, limite della fusion lato server) + giudice
  3. ricerca top-50 + famiglia (classificazione v2, family_classify.py) + giudice
Più la ricerca attuale senza giudice come riferimento.

Tutte le varianti usano lo stesso pool di ricerca (una sola chiamata con
limit=100 per caso; il top-50 è il suo inizio) e lo stesso giudice Liquid,
così le differenze dipendono solo dalla strategia. I giudizi già fatti da
realistic_eval.py per le stesse coppie vengono riusati.

Uso:
    python eval/realistic_compare.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import run_eval as r  # noqa: E402
import classify_archive as ca  # noqa: E402
import family_classify as fc  # noqa: E402
from judge_eval import judge_liquid, load_dotenv, rank_metrics, mean  # noqa: E402
from realistic_eval import CAND_DIR, REAL_DIR, RESULTS_DIR, safe, search_pool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=r.DEFAULT_BASE_URL)
    ap.add_argument("--wide", type=int, default=100)
    ap.add_argument("--model", default="d1.2-ultra")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_dotenv(EVAL_DIR / ".env")

    tcs = json.loads(r.DEFAULT_TEST_CASES.read_text(encoding="utf-8"))
    fam_rows = {row["key"]: row for row in fc.load_rows() if row["family"]}
    session = requests.Session()
    base = args.base_url.rstrip("/")
    report: Dict[str, Any] = {"cases": []}

    for tc in tcs:
        inp = tc["input"]
        mf = json.loads((CAND_DIR / inp / "manifest.json").read_text(encoding="utf-8"))
        query_img = CAND_DIR / inp / "masked" / mf["query_image"]
        case_dir = REAL_DIR / inp
        wide_dir = case_dir / f"wide{args.wide}"
        wide_dir.mkdir(parents=True, exist_ok=True)
        input_key = ca.archive_key(inp)
        print(f"\n[{inp}]", flush=True)

        pool = search_pool(session, base, query_img, args.wide, wide_dir)
        pool_c = [p for p in pool if p["key"] != input_key and p["image"]]
        top50 = [p for p in pool_c if p["rank"] <= 50]
        print(f"   ricerca: {len(pool)} risultati ({len(pool_c)} candidati con immagine)")

        fam_file = case_dir / "query_family_v2.json"
        if fam_file.exists():
            qfam = json.loads(fam_file.read_text(encoding="utf-8"))
        else:
            qfam = fc.classify_one(query_img, args.model)
            fam_file.write_text(json.dumps(qfam), encoding="utf-8")
        family = qfam["family"]
        true_fam = (fam_rows.get(input_key) or {}).get("family")
        klass = [
            {"key": k, "image": str(row["img"])}
            for k, row in fam_rows.items()
            if row["family"] == family and k != input_key
        ]
        print(f"   famiglia riconosciuta: {family} (p={qfam['top2'][0][1]:.2f}); in archivio: {true_fam}; classe: {len(klass)} disegni")

        judge_dir = case_dir / "judge"
        judge_dir.mkdir(exist_ok=True)
        todo = {}
        for c in pool_c + klass:
            f = judge_dir / f"{safe(c['key'])}.json"
            if not f.exists() and c["key"] not in todo:
                todo[c["key"]] = (Path(c["image"]), f)
        print(f"   coppie nuove da giudicare: {len(todo)}", flush=True)
        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(judge_liquid, query_img, img, args.model): f for img, f in todo.values()}
            for fu in as_completed(futs):
                try:
                    futs[fu].write_text(json.dumps(fu.result()), encoding="utf-8")
                except Exception as exc:
                    errors += 1
                    print(f"   errore giudice: {exc}")

        def p_of(key: str) -> float:
            f = judge_dir / f"{safe(key)}.json"
            return float(json.loads(f.read_text(encoding="utf-8")).get("similar", 0.0)) if f.exists() else -1.0

        def uniq(keys: List[str]) -> List[str]:
            seen, out = set(), []
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    out.append(k)
            return out

        k50 = uniq([p["key"] for p in top50])
        kwide = uniq([p["key"] for p in pool_c])
        kfam = uniq(k50 + [c["key"] for c in klass])
        variants = {
            "0_ricerca_50": k50,
            "1_ricerca50+giudice": sorted(k50, key=p_of, reverse=True),
            f"2_ricerca{args.wide}+giudice": sorted(kwide, key=p_of, reverse=True),
            "3_ricerca50+famiglia+giudice": sorted(kfam, key=p_of, reverse=True),
        }
        row = {"input": inp, "pool_size": len(pool), "family_recognized": family, "family_archive": true_fam,
               "class_size": len(klass), "judged_pairs": {k: len(v) for k, v in variants.items()}, "variants": {}}
        for vname, keys in variants.items():
            m = rank_metrics(tc, keys)
            row["variants"][vname] = m
            print(f"   {vname:>30} ({len(keys):>3} giudicati): attesi[{r.fmt_ranks(m['expected_ranks'])}]  "
                  f"indesiderati[{r.fmt_ranks(m['unwanted_ranks'])}]")
        report["cases"].append(row)

    print("\n=== Riepilogo (input senza cartiglio) ===")
    summary = {}
    for vname in report["cases"][0]["variants"]:
        rows = [c["variants"][vname] for c in report["cases"] if c["variants"][vname]["recall@10"] is not None]
        pairs = [c["judged_pairs"][vname] for c in report["cases"]]
        summary[vname] = {
            "recall@5": mean([x["recall@5"] for x in rows]),
            "recall@10": mean([x["recall@10"] for x in rows]),
            "recall@20": mean([x["recall@20"] for x in rows]),
            "mrr": mean([x["mrr"] for x in rows]),
            "unwanted@10": sum(c["variants"][vname]["unwanted@10"] for c in report["cases"]),
            "coppie_medie": round(sum(pairs) / len(pairs)),
        }
        print(f"{vname:>30}: " + "  ".join(f"{k}={v}" for k, v in summary[vname].items()))
    report["summary"] = summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"realistic_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRisultati salvati in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
