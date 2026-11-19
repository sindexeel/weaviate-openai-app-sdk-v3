"""
Caption libera vs caption strutturata: stabilità e correttezza (input senza cartiglio).

Stabilità (stesso disegno, N ripetizioni):
  - caption libera attuale (prompt di serve.py, gpt-4.1-mini): quanto cambia il testo
  - ricerca attuale sul server: quanto cambiano i risultati (top-10 / top-50)
  - caption strutturata Liquid (8 attributi di classify_archive.py)
  - caption strutturata GPT (stesso vocabolario, risposta JSON con valori fissi)

Correttezza: archivio ordinato per somiglianza degli attributi strutturati
(query Liquid vs attributi Liquid dell'archivio, già calcolati) confrontato
con la ricerca attuale, sugli stessi casi di test.

Niente scritture su Weaviate: solo /upload-image e /image-search.

Uso:
    python eval/caption_stability.py --reps 5
"""

from __future__ import annotations

import argparse
import ast
import base64
import itertools
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import run_eval as r  # noqa: E402
import classify_archive as ca  # noqa: E402
from judge_eval import load_dotenv, mean, rank_metrics  # noqa: E402

CAND_DIR = EVAL_DIR / "candidates"
WORK = CAND_DIR / "_capstab"
RESULTS_DIR = EVAL_DIR / "results"
SERVE = EVAL_DIR.parent / "serve.py"
CHOICE_Q = [q for q, d in ca.QUESTIONS.items() if d["type"] == "choice"]
BOOL_Q = [q for q, d in ca.QUESTIONS.items() if d["type"] == "noul"]


# ---------------------------------------------------------------------------
# caption libera: stessa funzione di serve.py, estratta dal sorgente
# ---------------------------------------------------------------------------

def load_free_caption_fn():
    from openai import OpenAI

    src = SERVE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "describe_mechanical_part")
    ns: Dict[str, Any] = {"_OPENAI_CLIENT": OpenAI(), "time": time}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SERVE), "exec"), ns)
    return ns["describe_mechanical_part"]


# ---------------------------------------------------------------------------
# caption strutturata GPT: stesso vocabolario delle domande Liquid
# ---------------------------------------------------------------------------

def gpt_structured(img: Path) -> Dict[str, Any]:
    from openai import OpenAI

    props: Dict[str, Any] = {}
    lines = []
    for q in CHOICE_Q:
        d = ca.QUESTIONS[q]
        props[q] = {"type": "string", "enum": list(d["criteria"])}
        lines.append(f"- {q}: {d['instructions']} " + "; ".join(f"{k} = {v}" for k, v in d["criteria"].items()))
    for q in BOOL_Q:
        props[q] = {"type": "boolean"}
        lines.append(f"- {q}: {ca.QUESTIONS[q]['instructions']}")
    schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    b64 = base64.b64encode(img.read_bytes()).decode("ascii")
    client = OpenAI()
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0,
                response_format={"type": "json_schema", "json_schema": {"name": "pezzo", "strict": True, "schema": schema}},
                messages=[
                    {"role": "system", "content": ca.STATE_TEXT},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Classifica il pezzo rispondendo a ogni campo:\n" + "\n".join(lines)},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]},
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            if "429" in str(exc):
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("troppi 429")


def liquid_flat(a: Dict[str, Any]) -> Dict[str, Any]:
    out = {q: ca.ch(a, q) for q in CHOICE_Q}
    out.update({q: (a.get(q) or 0) >= 0.5 for q in BOOL_Q})
    return out


# ---------------------------------------------------------------------------
# ricerca server (sola lettura)
# ---------------------------------------------------------------------------

def server_search(base: str, img: Path, limit: int) -> List[str]:
    s = requests.Session()
    with img.open("rb") as fh:
        up = s.post(f"{base}/upload-image", files={"image": (img.name, fh, "image/jpeg")}, timeout=600)
    up.raise_for_status()
    sr = s.post(f"{base}/image-search", json={"collection": "Sinde5", "image_id": up.json()["image_id"], "limit": limit}, timeout=600)
    sr.raise_for_status()
    return [ca.archive_key(r.result_name(x)) for x in sr.json().get("results") or []]


# ---------------------------------------------------------------------------
# metriche
# ---------------------------------------------------------------------------

def words(t: str) -> set:
    return set(re.findall(r"[a-zàèéìòù]+", t.lower()))


def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def pairwise(items: List[Any], f) -> float:
    pairs = list(itertools.combinations(items, 2))
    return round(sum(f(a, b) for a, b in pairs) / len(pairs), 3) if pairs else 1.0


def same_fields(dicts: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per ogni campo: frazione di ripetizioni uguali al valore più frequente."""
    out = {}
    for q in CHOICE_Q + BOOL_Q:
        vals = [d.get(q) for d in dicts]
        top = max(vals.count(v) for v in vals)
        out[q] = round(top / len(vals), 2)
    return out


def attr_score(q: Dict[str, Any], a: Dict[str, Any]) -> float:
    """Somiglianza morbida: prodotto delle probabilità per le scelte, 1-|Δp| per i sì/no."""
    s = 0.0
    for name in CHOICE_Q:
        pq = (q.get(name) or {}).get("probabilities") or {}
        pa = (a.get(name) or {}).get("probabilities") or {}
        s += sum(pq.get(k, 0) * pa.get(k, 0) for k in pq)
    for name in BOOL_Q:
        if q.get(name) is not None and a.get(name) is not None:
            s += 1 - abs(q[name] - a[name])
    return s


def attr_exact(q: Dict[str, Any], a: Dict[str, Any]) -> float:
    """Somiglianza 'da parole esatte' (come farebbe BM25 su una caption strutturata)."""
    fq, fa = liquid_flat(q), liquid_flat(a)
    w = {"part_type": 3, "shape_class": 2}
    return sum(w.get(k, 1) for k in fq if fq[k] == fa[k])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=r.DEFAULT_BASE_URL)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--model", default="d1.2-ultra")
    args = ap.parse_args()
    load_dotenv(EVAL_DIR / ".env")
    base = args.base_url.rstrip("/")

    tcs = json.loads(r.DEFAULT_TEST_CASES.read_text(encoding="utf-8"))
    free_caption = load_free_caption_fn()
    archive = {k: v[0] for k, v in ca.load_attrs().items()}
    print(f"archivio con attributi: {len(archive)} codici")
    report: Dict[str, Any] = {"cases": []}

    for tc in tcs:
        inp = tc["input"]
        mf = json.loads((CAND_DIR / inp / "manifest.json").read_text(encoding="utf-8"))
        img = CAND_DIR / inp / "masked" / mf["query_image"]
        d = WORK / inp
        d.mkdir(parents=True, exist_ok=True)
        input_key = ca.archive_key(inp)
        print(f"\n[{inp}]", flush=True)

        def cached(name: str, fn):
            f = d / f"{name}.json"
            if f.exists():
                return json.loads(f.read_text(encoding="utf-8"))
            v = fn()
            f.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
            return v

        jobs = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i in range(args.reps):
                jobs[("free", i)] = ex.submit(cached, f"free_{i}", lambda: free_caption(base64.b64encode(img.read_bytes()).decode("ascii")))
                jobs[("liq", i)] = ex.submit(cached, f"liquid_{i}", lambda: ca.classify(img, args.model))
                jobs[("gpt", i)] = ex.submit(cached, f"gptstruct_{i}", lambda: gpt_structured(img))
            # ricerche server in serie: non carichiamo il server di produzione
            searches = [cached(f"search_{i}", lambda: server_search(base, img, 50)) for i in range(args.reps)]
            res = {k: f.result() for k, f in jobs.items()}

        free = [res[("free", i)] for i in range(args.reps)]
        liq = [res[("liq", i)] for i in range(args.reps)]
        gpt = [res[("gpt", i)] for i in range(args.reps)]
        searches = [[k for k in s if k != input_key] for s in searches]

        stab = {
            "free_identiche": round(max(free.count(t) for t in free) / len(free), 2),
            "free_parole_jaccard": pairwise([words(t) for t in free], jacc),
            "free_prima_frase_uguale": round(max([t.split("—")[0].strip().lower() for t in free].count(x) for x in [t.split("—")[0].strip().lower() for t in free]) / len(free), 2),
            "ricerca_top10_jaccard": pairwise([set(s[:10]) for s in searches], jacc),
            "ricerca_top50_jaccard": pairwise([set(s[:50]) for s in searches], jacc),
            "liquid_campi_stabili": same_fields([liquid_flat(x) for x in liq]),
            "gpt_campi_stabili": same_fields(gpt),
        }
        stab["liquid_tutto_identico"] = round(max([json.dumps(liquid_flat(x), sort_keys=True) for x in liq].count(y) for y in [json.dumps(liquid_flat(x), sort_keys=True) for x in liq]) / len(liq), 2)
        stab["gpt_tutto_identico"] = round(max([json.dumps(x, sort_keys=True) for x in gpt].count(y) for y in [json.dumps(x, sort_keys=True) for x in gpt]) / len(gpt), 2)

        # correttezza
        q = liq[0]
        others = [(k, a) for k, a in archive.items() if k != input_key]
        rank_soft = [k for k, _ in sorted(others, key=lambda ka: attr_score(q, ka[1]), reverse=True)]
        rank_exact = [k for k, _ in sorted(others, key=lambda ka: attr_exact(q, ka[1]), reverse=True)]
        m_search = [rank_metrics(tc, s) for s in searches]
        m_soft = rank_metrics(tc, rank_soft)
        m_exact = rank_metrics(tc, rank_exact)
        # quanti disegni hanno lo stesso punteggio esatto del migliore (= "pareggi")
        best = attr_exact(q, archive[rank_exact[0]])
        ties = sum(1 for k, a in others if attr_exact(q, a) == best)

        row = {
            "input": inp,
            "query_struct_liquid": liquid_flat(q),
            "query_struct_gpt": gpt[0],
            "expected_struct": {c: liquid_flat(archive[r.code_key(c)]) if r.code_key(c) in archive else None for c in tc["expected"]},
            "stabilita": stab,
            "ricerca_attuale_per_run": [{"r10": m["recall@10"], "r20": m["recall@20"], "ranks": m["expected_ranks"]} for m in m_search],
            "strutturata_morbida": m_soft,
            "strutturata_esatta": m_exact,
            "pareggi_al_primo_posto_esatta": ties,
            "free_captions": [t[:160] for t in free],
        }
        report["cases"].append(row)
        print(f"   caption libera: identiche {stab['free_identiche']}, parole in comune {stab['free_parole_jaccard']}, prima frase uguale {stab['free_prima_frase_uguale']}")
        print(f"   ricerca attuale: top10 in comune {stab['ricerca_top10_jaccard']}, top50 {stab['ricerca_top50_jaccard']}; r10 per run {[m['recall@10'] for m in m_search]}")
        print(f"   strutturata Liquid tutto identico {stab['liquid_tutto_identico']}, GPT {stab['gpt_tutto_identico']}")
        print(f"   query Liquid: {liquid_flat(q)}")
        for c, s in row["expected_struct"].items():
            print(f"      atteso {c}: {s}")
        print(f"   attesi rank: morbida {r.fmt_ranks(m_soft['expected_ranks'])}  esatta {r.fmt_ranks(m_exact['expected_ranks'])} (pareggi in testa: {ties})")

    def avg(path):
        vals = [path(c) for c in report["cases"]]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "free_identiche": avg(lambda c: c["stabilita"]["free_identiche"]),
        "free_parole_jaccard": avg(lambda c: c["stabilita"]["free_parole_jaccard"]),
        "ricerca_top10_jaccard": avg(lambda c: c["stabilita"]["ricerca_top10_jaccard"]),
        "ricerca_top50_jaccard": avg(lambda c: c["stabilita"]["ricerca_top50_jaccard"]),
        "liquid_tutto_identico": avg(lambda c: c["stabilita"]["liquid_tutto_identico"]),
        "gpt_tutto_identico": avg(lambda c: c["stabilita"]["gpt_tutto_identico"]),
        "ricerca_r10_media": avg(lambda c: mean([x["r10"] for x in c["ricerca_attuale_per_run"]])),
        "ricerca_r10_min": avg(lambda c: min(x["r10"] for x in c["ricerca_attuale_per_run"])),
        "ricerca_r10_max": avg(lambda c: max(x["r10"] for x in c["ricerca_attuale_per_run"])),
        "ricerca_r20_media": avg(lambda c: mean([x["r20"] for x in c["ricerca_attuale_per_run"]])),
        "strutt_morbida_r10": avg(lambda c: c["strutturata_morbida"]["recall@10"]),
        "strutt_morbida_r20": avg(lambda c: c["strutturata_morbida"]["recall@20"]),
        "strutt_esatta_r10": avg(lambda c: c["strutturata_esatta"]["recall@10"]),
        "strutt_esatta_r20": avg(lambda c: c["strutturata_esatta"]["recall@20"]),
    }
    report["summary"] = summary
    print("\n=== Riepilogo ===")
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"caption_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRisultati salvati in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
