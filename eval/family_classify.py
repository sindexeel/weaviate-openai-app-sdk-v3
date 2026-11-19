"""
Classificazione preliminare dell'archivio in famiglie di pezzo, da mostrare
al cliente e da fargli validare.

Le famiglie partono dai prefissi che il cliente/fornitori usano già nei nomi
file (FLANG, DISTA, ALBER, ...). Per ogni disegno:
  1. se il nome file ha un prefisso noto  -> famiglia dal prefisso (fonte certa)
  2. altrimenti                           -> famiglia proposta da Liquid
                                             (guarda il disegno intero, cartiglio compreso:
                                             per l'archivio è un'informazione legittima)
Liquid classifica comunque TUTTI i disegni: sui disegni con prefisso si misura
quanto è affidabile.

Output (in eval/candidates/_archive/, ignorato da git, contiene i disegni del cliente):
  - famiglie_archivio.csv      : un disegno per riga, da far correggere al cliente
  - famiglie_report.html       : riepilogo con esempi per famiglia

Uso:
    python eval/family_classify.py classify
    python eval/family_classify.py report
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import classify_archive as ca  # noqa: E402
from judge_eval import load_dotenv  # noqa: E402

OUT_DIR = ca.WORK
FAM_DIR = OUT_DIR / "families_v2"

# famiglia -> (descrizione per il modello, prefissi del cliente)
FAMILIES: Dict[str, tuple] = {
    "albero": ("albero o mandrino: pezzo tornito lungo con più diametri, sedi, cave", ["ALBER", "MANDR"]),
    "tirante_barra": ("tirante, barra, asta: elemento lungo e sottile, spesso filettato alle estremità", ["TIRAN", "BARRA", "ASTA"]),
    "perno_spina": ("perno, spina, piolo, codolo: tornito corto e pieno", ["PERNO", "SPINA", "PIOLO", "CODOL"]),
    "canotto_tubo": ("canotto, tubo, manicotto: tornito cavo e lungo", ["CANOT", "TUBOL", "TUBO", "MANIC"]),
    "boccola_bussola": ("boccola o bussola: anello tornito cavo con pareti spesse, spesso con collare", ["BUSSO", "BOCCO", "BUSRI"]),
    "distanziale_anello": ("distanziale, anello, rondella, spessore: tornito corto, cavo, sottile", ["DISTA", "ANELL", "RONDE", "SPESS"]),
    "flangia_disco": ("flangia o disco: tornito piatto con foratura sulla circonferenza", ["FLANG", "DISCO"]),
    "coperchio_tappo": ("coperchio o tappo di chiusura", ["COPER", "TAPPO"]),
    "mozzo_volantino": ("mozzo o volantino: corpo tornito con foro centrale per albero", ["MOZZO", "VOLAN"]),
    "puleggia_rullo": ("puleggia, carrucola, rullo: tornito con gole o superficie di rotolamento", ["PULEG", "RULLO"]),
    "ingranaggio_camma": ("ingranaggio, pignone, ruota dentata, camma", ["INGRA", "CAMME", "CAMPA"]),
    "ghiera_vite": ("ghiera, dado, chiocciola, vite speciale: elemento filettato", ["GHIER", "VITSP", "CHIOC"]),
    "blocchetto_corpo": ("blocchetto o corpo prismatico massiccio fresato, portapezzi, tassello", ["BLOCC", "PORTA", "TASSE"]),
    "piastra": ("piastra o piatto fresato, spessore costante", ["PIAST", "PIATT"]),
    "supporto_staffa": ("supporto, staffa, squadra, mensola, forcella", ["SUPPO", "STAFF", "SUPCH", "SQUAD", "MENSO", "FORCE"]),
    "leva": ("leva o braccio", ["LEVA"]),
    "lardone_guida": ("lardone, pattino, guida di scorrimento", ["LARDO", "PATTI"]),
    "lamiera_carpenteria": ("lamiera piegata o tagliata, carter, protezione, carpenteria saldata", []),
    "nipplo_raccordo": ("nipplo o raccordo per fluidi", []),
    "altro": ("nessuna delle precedenti", []),
}
PREFIX_TO_FAMILY = {p: fam for fam, (_, prefs) in FAMILIES.items() for p in prefs}
_PREFIX_RE = re.compile(r"^([A-Z]{4,5})0*\d", re.I)


def prefix_of(name: str) -> Optional[str]:
    m = _PREFIX_RE.match(name)
    if not m or re.match(r"^(PF|SM|P0)", name, re.I):
        return None
    return m.group(1).upper()


def classify_one(img: Path, model: str) -> Dict[str, Any]:
    body = {
        "model": model,
        "state": "Disegno tecnico di un componente meccanico.",
        "images": ["data:image/jpeg;base64," + base64.b64encode(img.read_bytes()).decode("ascii")],
        "questions": {
            "family": {
                "type": "choice",
                "instructions": "A quale famiglia di pezzo appartiene il componente del disegno?",
                "criteria": {k: v[0] for k, v in FAMILIES.items()},
            }
        },
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
        a = resp.json()["answers"]["family"]
        probs = a.get("probabilities") or {}
        if isinstance(probs, list):
            probs = {p.get("value") or p.get("label"): p.get("probability") for p in probs}
        top2 = sorted(probs.items(), key=lambda x: -(x[1] or 0))[:2]
        return {"family": a.get("choice"), "confidence": a.get("confidence"), "top2": top2,
                "_input_tokens": resp.json().get("usage", {}).get("input_tokens")}
    raise RuntimeError("troppi tentativi")


def archive_items() -> List[Dict[str, Any]]:
    """Un elemento per codice distinto, con immagine già renderizzata da classify_archive.py."""
    items: Dict[str, Dict[str, Any]] = {}
    for f in ca.list_archive():
        key = ca.archive_key(f.name)
        if key in items:
            continue
        img = ca.IMG_DIR / f"{ca.file_id(f)}.jpg"
        if not img.exists():
            continue
        rel = f.relative_to(ca.PROJECT).as_posix()
        source = "wetransfer" if rel.startswith("wetransfer") else rel.split("/")[2]
        items[key] = {"key": key, "name": f.name, "file": rel, "source": source, "img": img, "prefix": prefix_of(f.stem)}
    return list(items.values())


def cmd_classify(args) -> int:
    items = archive_items()
    FAM_DIR.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (FAM_DIR / f"{ca.file_id(ca.PROJECT / it['file'])}.json").exists()]
    print(f"Disegni: {len(items)}, da classificare: {len(todo)}")
    tokens = errors = 0

    def work(it):
        res = classify_one(it["img"], args.model)
        (FAM_DIR / f"{ca.file_id(ca.PROJECT / it['file'])}.json").write_text(json.dumps(res), encoding="utf-8")
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                tokens += fu.result().get("_input_tokens") or 0
            except Exception as exc:
                errors += 1
                print(f"   errore: {exc}")
            if i % 250 == 0:
                print(f"   {i}/{len(todo)}", flush=True)
    print(f"Fatto. Errori: {errors}, token {tokens}, costo ~${tokens / 1e6 * 0.042:.3f}")
    return 0


def load_rows() -> List[Dict[str, Any]]:
    rows = []
    for it in archive_items():
        f = FAM_DIR / f"{ca.file_id(ca.PROJECT / it['file'])}.json"
        liq = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        pref_fam = PREFIX_TO_FAMILY.get(it["prefix"]) if it["prefix"] else None
        if pref_fam:
            final, src, conf = pref_fam, "prefisso", 1.0
        else:
            top = (liq.get("top2") or [[None, None]])[0]
            # probabilità della famiglia scelta (stessa scala dell'alternativa nel CSV)
            final, src, conf = liq.get("family"), "Liquid", top[1]
        rows.append({**it, "liquid": liq.get("family"), "liquid_conf": liq.get("confidence"),
                     "liquid_top2": liq.get("top2"), "prefix_family": pref_fam,
                     "family": final, "family_source": src, "family_conf": conf})
    return rows


def thumb_b64(img: Path, side: int = 260) -> str:
    im = Image.open(img).convert("L")
    im.thumbnail((side, side))
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def cmd_report(args) -> int:
    rows = load_rows()
    rows = [r for r in rows if r["family"]]
    n = len(rows)

    # affidabilità di Liquid dove il prefisso dà la risposta
    checked = [r for r in rows if r["prefix_family"] and r["liquid"]]
    ok = sum(1 for r in checked if r["liquid"] == r["prefix_family"])
    per_fam = defaultdict(lambda: [0, 0])
    confusions = Counter()
    for r in checked:
        per_fam[r["prefix_family"]][1] += 1
        if r["liquid"] == r["prefix_family"]:
            per_fam[r["prefix_family"]][0] += 1
        else:
            confusions[(r["prefix_family"], r["liquid"])] += 1
    print(f"Disegni classificati: {n}")
    print(f"Affidabilità Liquid sui disegni con prefisso: {ok}/{len(checked)} = {ok / max(len(checked), 1):.0%}")
    for fam, (a, b) in sorted(per_fam.items(), key=lambda x: -x[1][1]):
        print(f"   {fam:<22} {a}/{b}")
    print("Confusioni principali:", confusions.most_common(8))
    unknown_prefixes = Counter(r["prefix"] for r in rows if r["prefix"] and not r["prefix_family"])
    print("Prefissi senza famiglia (da chiarire col cliente):", unknown_prefixes.most_common())

    # CSV per il cliente
    csv_path = OUT_DIR / "famiglie_archivio.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["codice", "file", "cartella", "famiglia_proposta", "fonte", "confidenza", "alternativa", "famiglia_corretta_cliente", "note_cliente"])
        for r in sorted(rows, key=lambda r: (r["family"], r["key"])):
            alt = ""
            if r["family_source"] == "Liquid" and r["liquid_top2"]:
                others = [(k, p) for k, p in r["liquid_top2"] if k != r["family"]]
                if others:
                    alt = f"{others[0][0]} ({(others[0][1] or 0):.2f})"
            w.writerow([r["key"], r["file"], r["source"], r["family"], r["family_source"],
                        f"{(r['family_conf'] or 0):.2f}", alt, "", ""])
    print(f"CSV: {csv_path}")

    # HTML
    fams = Counter(r["family"] for r in rows)
    src_counts = Counter(r["family_source"] for r in rows)
    low = sum(1 for r in rows if r["family_source"] == "Liquid" and (r["family_conf"] or 0) < 0.5)
    parts = []
    for fam, cnt in fams.most_common():
        fr = [r for r in rows if r["family"] == fam]
        lav = Counter(r["source"] for r in fr).most_common(3)
        ex = sorted(fr, key=lambda r: -(r["family_conf"] or 0))[:6]
        thumbs = "".join(
            f'<figure><img src="data:image/jpeg;base64,{thumb_b64(r["img"])}"><figcaption>{html.escape(r["key"])}'
            f'<br><small>{r["family_source"]} {(r["family_conf"] or 0):.2f}</small></figcaption></figure>'
            for r in ex
        )
        desc = html.escape(FAMILIES.get(fam, ("",))[0])
        prefs = ", ".join(FAMILIES.get(fam, ("", []))[1]) or "—"
        acc = per_fam.get(fam)
        acc_txt = f"{acc[0]}/{acc[1]} ({acc[0] / acc[1]:.0%})" if acc and acc[1] else "n.d."
        parts.append(
            f'<section><h2>{html.escape(fam.replace("_", " / "))} <span class="n">{cnt} disegni · {cnt / n:.0%}</span></h2>'
            f'<p class="d">{desc}<br>Prefissi del cliente: <b>{prefs}</b> · Accordo Liquid/prefisso: <b>{acc_txt}</b>'
            f' · Cartelle: {", ".join(f"{s} {c}" for s, c in lav)}</p><div class="g">{thumbs}</div></section>'
        )
    page = f"""<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Famiglie archivio Sinde</title><style>
:root{{--bg:#fff;--fg:#1c1c1c;--muted:#666;--line:#ddd;--card:#f6f6f4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161616;--fg:#eee;--muted:#aaa;--line:#333;--card:#222}}}}
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:24px 16px;max-width:1200px;margin:auto}}
h1{{margin:0 0 4px}} .sub{{color:var(--muted);margin:0 0 20px}}
.kpi{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}} .kpi div{{background:var(--card);padding:12px 16px;border-radius:8px}}
.kpi b{{font-size:22px;display:block}} section{{border-top:1px solid var(--line);padding:16px 0}}
h2{{font-size:18px;margin:0 0 4px;text-transform:capitalize}} .n{{color:var(--muted);font-weight:400;font-size:14px;text-transform:none}}
.d{{color:var(--muted);font-size:13px;margin:0 0 10px}} .g{{display:flex;gap:10px;flex-wrap:wrap}}
figure{{margin:0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px;width:180px}}
figure img{{width:100%;height:140px;object-fit:contain}} figcaption{{font-size:12px;color:#333}}
.note{{background:var(--card);padding:12px 16px;border-radius:8px;font-size:14px}}
</style></head><body>
<h1>Famiglie dell'archivio disegni</h1>
<p class="sub">Classificazione preliminare automatica · da validare</p>
<div class="kpi"><div><b>{n}</b>disegni classificati</div><div><b>{len(fams)}</b>famiglie</div>
<div><b>{src_counts.get("prefisso", 0)}</b>da prefisso del nome</div><div><b>{src_counts.get("Liquid", 0)}</b>proposte dal modello</div>
<div><b>{ok / max(len(checked), 1):.0%}</b>accordo modello/prefisso ({len(checked)} disegni)</div><div><b>{low}</b>proposte incerte (&lt;0,5)</div></div>
<p class="note">Le famiglie partono dai prefissi già presenti nei nomi file. Dove il prefisso manca, la famiglia è proposta da un modello che guarda il disegno.
Il file <b>famiglie_archivio.csv</b> elenca tutti i disegni con la famiglia proposta: le colonne vuote servono per le vostre correzioni.</p>
{''.join(parts)}
</body></html>"""
    html_path = OUT_DIR / "famiglie_report.html"
    html_path.write_text(page, encoding="utf-8")
    print(f"HTML: {html_path} ({html_path.stat().st_size // 1024} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--model", default="d1.2-ultra")
    c.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    args = ap.parse_args()
    load_dotenv(EVAL_DIR / ".env")
    return cmd_classify(args) if args.cmd == "classify" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
