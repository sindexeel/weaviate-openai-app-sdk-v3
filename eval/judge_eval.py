"""
Confronto tra il ranking attuale e un "giudice" che valuta ogni coppia
(disegno di input, candidato) sui candidati congelati da freeze_candidates.py.

Giudici disponibili:
  liquid  - Liquid AI decision model (d1.2-ultra di default), probabilità calibrate
  llm     - LLM generico OpenAI (gpt-4.1-mini di default) come controllo, stesse domande

Il giudice riordina i candidati per P(simile); il confronto usa la stessa
ground truth di test_cases.json e le stesse metriche di run_eval.py, più
un'analisi a soglia ("mostra solo i candidati con P(simile) >= t").

Chiavi API in eval/.env (ignorato da git):
    LIQUID_API_KEY=...
    OPENAI_API_KEY=...

Uso:
    python eval/judge_eval.py --judge liquid --max-cases 1 --max-cands 5   # prova
    python eval/judge_eval.py --judge liquid
    python eval/judge_eval.py --judge llm
    python eval/judge_eval.py --report liquid llm                          # solo confronto da cache
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import run_eval as r  # noqa: E402

CAND_DIR = EVAL_DIR / "candidates"
CACHE_DIR = EVAL_DIR / "candidates" / "_judge_cache"  # dentro candidates/: ignorato da git
RESULTS_DIR = EVAL_DIR / "results"

STATE_TEXT = (
    "Due disegni tecnici di componenti meccanici. "
    "Immagine 1 = disegno di riferimento. Immagine 2 = disegno candidato. "
    "Valuta solo la geometria del pezzo: ignora cartiglio, tabelle, quote numeriche, testi e note."
)

QUESTIONS: Dict[str, str] = {
    "same_type": (
        "Il componente dell'immagine 2 è dello stesso tipo del componente dell'immagine 1 "
        "(es. entrambi pignoni/ingranaggi, entrambi alberi, entrambi staffe/supporti, "
        "entrambi raccordi/nippli, entrambi canotti/boccole, entrambi blocchetti)?"
    ),
    "same_shape_class": (
        "I due componenti hanno la stessa classe di forma: entrambi torniti/assialsimmetrici "
        "(sezioni circolari, diametri, simbolo Ø) oppure entrambi prismatici/fresati/lamiera?"
    ),
    "similar_proportions": (
        "I due componenti hanno proporzioni simili (rapporto lunghezza/diametro o "
        "lunghezza/larghezza/spessore confrontabile)?"
    ),
    "similar": (
        "Un tecnico che deve stimare il costo di produzione del pezzo dell'immagine 1 "
        "considererebbe il pezzo dell'immagine 2 un riferimento simile e utile "
        "(stessa tipologia, forma e proporzioni confrontabili, lavorazioni analoghe)?"
    ),
}


# ---------------------------------------------------------------------------
# utilità
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def load_cases() -> List[Dict[str, Any]]:
    tcs = {tc["input"]: tc for tc in json.loads(r.DEFAULT_TEST_CASES.read_text(encoding="utf-8"))}
    cases = []
    for inp, tc in tcs.items():
        mf = CAND_DIR / inp / "manifest.json"
        if not mf.exists():
            print(f"[{inp}] candidati non congelati: esegui freeze_candidates.py")
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        input_key = r.code_key(inp)
        cands = [c for c in m["candidates"] if c["key"] != input_key and c["image"]]
        cases.append({"tc": tc, "dir": CAND_DIR / inp, "query": m["query_image"], "cands": cands})
    return cases


# ---------------------------------------------------------------------------
# giudici
# ---------------------------------------------------------------------------

def judge_liquid(q_img: Path, c_img: Path, model: str) -> Dict[str, float]:
    key = os.environ.get("LIQUID_API_KEY")
    if not key:
        raise RuntimeError("LIQUID_API_KEY mancante (eval/.env)")
    body = {
        "model": model,
        "state": STATE_TEXT,
        "images": [data_url(q_img), data_url(c_img)],
        "questions": {name: {"type": "noul", "instructions": text} for name, text in QUESTIONS.items()},
    }
    for attempt in range(5):
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
        ans = resp.json().get("answers", {})
        out = {name: float(ans[name]["noul"]) for name in QUESTIONS if name in ans and "noul" in ans[name]}
        out["_input_tokens"] = resp.json().get("usage", {}).get("input_tokens")
        return out
    raise RuntimeError("troppi tentativi (429/5xx)")


def judge_llm(q_img: Path, c_img: Path, model: str) -> Dict[str, float]:
    from openai import OpenAI

    client = OpenAI()
    props = {name: {"type": "number", "minimum": 0, "maximum": 1} for name in QUESTIONS}
    schema = {
        "type": "object",
        "properties": props,
        "required": list(QUESTIONS),
        "additionalProperties": False,
    }
    questions_txt = "\n".join(f"- {name}: {text}" for name, text in QUESTIONS.items())
    prompt = (
        f"{STATE_TEXT}\n\nPer ciascuna domanda restituisci la probabilità (0-1) che la risposta sia sì.\n"
        f"{questions_txt}"
    )
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": {"name": "giudizio", "schema": schema, "strict": True}},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url(q_img), "detail": "high"}},
                        {"type": "image_url", "image_url": {"url": data_url(c_img), "detail": "high"}},
                    ],
                }],
            )
            out = json.loads(resp.choices[0].message.content)
            out["_input_tokens"] = getattr(resp.usage, "prompt_tokens", None)
            return out
        except Exception as exc:
            if "429" in str(exc) or "rate" in str(exc).lower():
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("troppi tentativi (rate limit)")


JUDGES = {"liquid": judge_liquid, "llm": judge_llm}
DEFAULT_MODELS = {"liquid": "d1.2-ultra", "llm": "gpt-4.1-mini"}


# ---------------------------------------------------------------------------
# metriche
# ---------------------------------------------------------------------------

def rank_metrics(tc: Dict[str, Any], ordered_keys: List[str]) -> Dict[str, Any]:
    def rank_of(code: str) -> Optional[int]:
        ck = r.code_key(code)
        return next((i + 1 for i, k in enumerate(ordered_keys) if k == ck), None)

    exp = {c: rank_of(c) for c in tc.get("expected", [])}
    unw = {c: rank_of(c) for c in tc.get("unwanted", [])}

    def rec(k: int) -> Optional[float]:
        return sum(1 for v in exp.values() if v and v <= k) / len(exp) if exp else None

    found = [v for v in exp.values() if v]
    return {
        "expected_ranks": exp,
        "unwanted_ranks": unw,
        "recall@5": rec(5),
        "recall@10": rec(10),
        "recall@20": rec(20),
        "mrr": (1.0 / min(found)) if found else 0.0,
        "unwanted@5": sum(1 for v in unw.values() if v and v <= 5),
        "unwanted@10": sum(1 for v in unw.values() if v and v <= 10),
    }


def threshold_metrics(tc: Dict[str, Any], scored: List[Dict[str, Any]], t: float) -> Dict[str, Any]:
    shown = [s for s in scored if s["p"] >= t]
    exp_keys = {r.code_key(c) for c in tc.get("expected", [])}
    unw_keys = {r.code_key(c) for c in tc.get("unwanted", [])}
    return {
        "shown": len(shown),
        "expected_shown": sum(1 for s in shown if s["key"] in exp_keys),
        "unwanted_shown": sum(1 for s in shown if s["key"] in unw_keys),
    }


def mean(vals: List[Optional[float]]) -> Optional[float]:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 3) if v else None


# ---------------------------------------------------------------------------
# esecuzione
# ---------------------------------------------------------------------------

def run_judge(judge: str, model: str, cases: List[Dict[str, Any]], max_cands: int, workers: int) -> None:
    fn = JUDGES[judge]
    tasks = []
    for case in cases:
        cache = CACHE_DIR / f"{judge}_{model}" / case["tc"]["input"]
        cache.mkdir(parents=True, exist_ok=True)
        for c in case["cands"][:max_cands]:
            out_file = cache / f"{c['image']}.json"
            if not out_file.exists():
                tasks.append((case["dir"] / case["query"], case["dir"] / c["image"], out_file, case["tc"]["input"], c["name"]))
    print(f"[{judge}:{model}] coppie da giudicare: {len(tasks)} (le altre sono in cache)")
    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, q, c, model): (out, inp, name) for q, c, out, inp, name in tasks}
        for f in as_completed(futs):
            out, inp, name = futs[f]
            try:
                res = f.result()
                out.write_text(json.dumps(res), encoding="utf-8")
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    print(f"   {done}/{len(tasks)}", flush=True)
            except Exception as exc:
                errors += 1
                print(f"   errore {inp} / {name}: {exc}")
    if errors:
        print(f"[{judge}] {errors} coppie in errore: rilanciare per completarle (la cache evita di ripetere le altre)")


def evaluate(judge_label: Optional[str], cases: List[Dict[str, Any]], max_cands: int, thresholds: List[float]) -> Dict[str, Any]:
    rows = []
    for case in cases:
        tc = case["tc"]
        cands = case["cands"][:max_cands]
        if judge_label is None:
            ordered = [c["key"] for c in cands]
            scored = None
        else:
            cache = CACHE_DIR / judge_label / tc["input"]
            scored = []
            for c in cands:
                f = cache / f"{c['image']}.json"
                if f.exists():
                    a = json.loads(f.read_text(encoding="utf-8"))
                    scored.append({**c, "p": float(a.get("similar", 0.0)), "answers": a})
            if len(scored) < len(cands):
                print(f"[{judge_label}] {tc['input']}: {len(cands) - len(scored)} coppie senza giudizio, escluse")
            # a parità di P(simile) conta l'ordine originale (sort stabile)
            scored.sort(key=lambda s: s["p"], reverse=True)
            ordered = [s["key"] for s in scored]
        m = rank_metrics(tc, ordered)
        m["input"] = tc["input"]
        if scored is not None:
            m["thresholds"] = {str(t): threshold_metrics(tc, scored, t) for t in thresholds}
            exp_keys = {r.code_key(c) for c in tc.get("expected", [])}
            unw_keys = {r.code_key(c) for c in tc.get("unwanted", [])}
            m["p_expected"] = [round(s["p"], 3) for s in scored if s["key"] in exp_keys]
            m["p_unwanted"] = [round(s["p"], 3) for s in scored if s["key"] in unw_keys]
            m["p_others_mean"] = mean([s["p"] for s in scored if s["key"] not in exp_keys | unw_keys])
            m["top5"] = [(s["name"], round(s["p"], 3)) for s in scored[:5]]
        rows.append(m)

    with_exp = [x for x in rows if x["recall@10"] is not None]
    summary = {
        "recall@5": mean([x["recall@5"] for x in with_exp]),
        "recall@10": mean([x["recall@10"] for x in with_exp]),
        "recall@20": mean([x["recall@20"] for x in with_exp]),
        "mrr": mean([x["mrr"] for x in with_exp]),
        "unwanted@5_total": sum(x["unwanted@5"] for x in rows),
        "unwanted@10_total": sum(x["unwanted@10"] for x in rows),
    }
    if judge_label is not None:
        for t in thresholds:
            ts = [x["thresholds"][str(t)] for x in rows]
            summary[f"t{t}_shown_avg"] = round(sum(s["shown"] for s in ts) / len(ts), 1) if ts else None
            summary[f"t{t}_expected_shown"] = sum(s["expected_shown"] for s in ts)
            summary[f"t{t}_unwanted_shown"] = sum(s["unwanted_shown"] for s in ts)
        summary["p_expected_mean"] = mean([p for x in rows for p in x["p_expected"]])
        summary["p_unwanted_mean"] = mean([p for x in rows for p in x["p_unwanted"]])
        summary["p_others_mean"] = mean([x["p_others_mean"] for x in rows])
    return {"summary": summary, "cases": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=sorted(JUDGES))
    ap.add_argument("--model", help="modello del giudice (default: d1.2-ultra / gpt-4.1-mini)")
    ap.add_argument("--max-cases", type=int, default=0, help="0 = tutti")
    ap.add_argument("--max-cands", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument("--report", nargs="*", metavar="JUDGE", help="solo report da cache, es. --report liquid llm")
    args = ap.parse_args()

    load_dotenv(EVAL_DIR / ".env")
    cases = load_cases()
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        return 1

    judges = args.report if args.report is not None else ([args.judge] if args.judge else [])
    labels: List[str] = []
    for j in judges:
        model = args.model or DEFAULT_MODELS[j]
        label = f"{j}_{model}"
        if args.report is None:
            run_judge(j, model, cases, args.max_cands, args.workers)
        labels.append(label)

    report: Dict[str, Any] = {"baseline": evaluate(None, cases, args.max_cands, args.thresholds)}
    for label in labels:
        report[label] = evaluate(label, cases, args.max_cands, args.thresholds)

    keys = list(report["baseline"]["summary"].keys())
    print("\n=== Confronto (stessi candidati congelati, top-%d) ===" % args.max_cands)
    header = f"{'metrica':>24}" + "".join(f"{name[:22]:>24}" for name in report)
    print(header)
    all_keys = keys + [k for lab in labels for k in report[lab]["summary"] if k not in keys]
    seen = set()
    for k in all_keys:
        if k in seen:
            continue
        seen.add(k)
        print(f"{k:>24}" + "".join(f"{str(report[n]['summary'].get(k, '')):>24}" for n in report))

    for label in labels:
        print(f"\n--- {label}: dettaglio per caso ---")
        for c in report[label]["cases"]:
            print(f"{c['input']:>15}  attesi[{r.fmt_ranks(c['expected_ranks'])}]  indesiderati[{r.fmt_ranks(c['unwanted_ranks'])}]  "
                  f"P(attesi)={c['p_expected']} P(indes.)={c['p_unwanted']} P(altri)~{c['p_others_mean']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"judges_{'_'.join(labels) or 'baseline'}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRisultati salvati in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
