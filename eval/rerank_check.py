"""
Verifica in locale il riordino Liquid di serve.py (_liquid_rerank) senza
avviare il server: usa le query senza cartiglio e i primi 50 candidati già
salvati da realistic_compare.py (candidates/_real/<caso>/wide100).

Uso:
    python eval/rerank_check.py
"""

from __future__ import annotations

import ast
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import run_eval as r  # noqa: E402
import classify_archive as ca  # noqa: E402
from judge_eval import load_dotenv, mean, rank_metrics  # noqa: E402

SERVE = EVAL_DIR.parent / "serve.py"
CAND_DIR = EVAL_DIR / "candidates"
REAL_DIR = CAND_DIR / "_real"

RERANK_NAMES = {
    "_LIQUID_API_URL", "_LIQUID_MODEL", "_LIQUID_RERANK_POOL", "_LIQUID_WORKERS", "_LIQUID_TIMEOUT",
    "_LIQUID_MIN_P", "_LIQUID_MAX_SIDE", "_LIQUID_STATE", "_LIQUID_QUESTIONS",
    "_liquid_image_url", "_liquid_judge_pair", "_liquid_rerank",
}


def load_rerank_module() -> Dict[str, Any]:
    """Esegue solo le definizioni del riordino prese da serve.py (niente Weaviate/MCP)."""
    tree = ast.parse(SERVE.read_text(encoding="utf-8"))
    keep = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in RERANK_NAMES:
            keep.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in RERANK_NAMES for t in n.targets):
            keep.append(n)
    ns: Dict[str, Any] = {"os": os, "time": time, "base64": base64, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SERVE), "exec"), ns)
    return ns


def main() -> int:
    load_dotenv(EVAL_DIR / ".env")
    ns = load_rerank_module()
    tcs = json.loads(r.DEFAULT_TEST_CASES.read_text(encoding="utf-8"))
    rows = []
    for tc in tcs:
        inp = tc["input"]
        mf = json.loads((CAND_DIR / inp / "manifest.json").read_text(encoding="utf-8"))
        q_b64 = base64.b64encode((CAND_DIR / inp / "masked" / mf["query_image"]).read_bytes()).decode("ascii")
        input_key = ca.archive_key(inp)
        pool = json.loads((REAL_DIR / inp / "wide100" / "pool.json").read_text(encoding="utf-8"))
        top = [p for p in pool if p["rank"] <= 50 and p["image"] and p["key"] != input_key]
        items = [{"source_pdf": p["name"], "image_b64": base64.b64encode(Path(p["image"]).read_bytes()).decode("ascii")} for p in top]
        t0 = time.time()
        judged = ns["_liquid_rerank"](q_b64, items)
        dt = time.time() - t0
        before = [p["key"] for p in top]
        if judged is None:
            print(f"[{inp}] riordino non disponibile")
            continue
        order = sorted(top, key=lambda p: (judged.get(p["name"]) or {}).get("similar", -1.0), reverse=True)
        after = [p["key"] for p in order]
        mb, ma = rank_metrics(tc, before), rank_metrics(tc, after)
        rows.append((mb, ma, dt))
        print(f"[{inp}] {len(judged)}/{len(items)} giudicati in {dt:.1f}s")
        print(f"   prima: attesi[{r.fmt_ranks(mb['expected_ranks'])}] indesiderati[{r.fmt_ranks(mb['unwanted_ranks'])}]")
        print(f"   dopo : attesi[{r.fmt_ranks(ma['expected_ranks'])}] indesiderati[{r.fmt_ranks(ma['unwanted_ranks'])}]")
    if rows:
        for label, i in (("prima", 0), ("dopo", 1)):
            print(f"{label:>6}: r5={mean([x[i]['recall@5'] for x in rows])} r10={mean([x[i]['recall@10'] for x in rows])} "
                  f"r20={mean([x[i]['recall@20'] for x in rows])} mrr={mean([x[i]['mrr'] for x in rows])}")
        print(f"tempo medio riordino: {sum(x[2] for x in rows) / len(rows):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
