"""
report.py – Benchmark completo: 5 modelli × 6 explainer × 3 metriche (cap. 3-4)
===============================================================================

Esempi
------
    # stima dei tempi sull'hardware in uso (pochi minuti), da lanciare prima del run
    python report.py --estimate

    # run completo come nella tesi (400 testi, tutte le metriche)
    python report.py --out-dir runs/v2

    # solo alcuni modelli/explainer, per esempio Attention Flow su un runtime CPU
    python report.py --explainers attention_flow --device cpu --flow-workers 2

Il run si può interrompere e rilanciare con lo stesso comando: ogni testo
completato è salvato subito (file .jsonl) e viene saltato alla ripresa.

Esecuzione in parallelo: `--shard I/N` fa elaborare a questo processo solo i
testi con indice i % N == I, in file separati (eval.s{I}of{N}.jsonl). Più
processi possono lavorare sulla stessa cartella; i risultati non dipendono dalla
divisione, perché ogni testo ha i propri seed. Alla fine `--summarize-only`
ricalcola riepiloghi e tabelle da tutti i file. parallel_run.py organizza tutto.

Per ogni coppia (modello, explainer):
1. modello in eval(), seed 42: spiegazione del testo originale e delle 6
   perturbazioni (robustness); la spiegazione originale serve anche alla contrastivity;
2. modello in train() (dropout attivo): 4 spiegazioni con seed 42, 123, 456, 789
   (consistency).

Output in --out-dir:
    <modello>/<explainer>/eval*.jsonl, train*.jsonl  valori per testo
    <modello>/<explainer>/summary.json               metriche aggregate e conteggi
    robustness_table.csv, consistency_table.csv, contrastivity_table.csv
    run_metadata/*.json                              configurazione, versioni, commit (uno per processo)
"""

import argparse
import hashlib
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import dataset
import metrics
from attribution import Attribution, normalize_whitespace
from utils import Timer, append_jsonl, format_time, git_commit, read_jsonl, save_json, set_seed

EXPLAINERS = ["lime", "shap", "grad_input", "attention_rollout", "attention_flow", "lrp"]
METRICS = ["robustness", "consistency", "contrastivity"]
MODEL_ORDER = ["tinybert", "distilbert", "roberta-base", "bert-large", "roberta-large"]
FAIL_FAST_AFTER = 3  # errori consecutivi iniziali dopo cui un explainer viene fermato
MASK_STRINGS = ("[MASK]", "<mask>")  # parole escluse dal vocabolario della contrastivity


# =============================================================================
# Preparazione dei testi
# =============================================================================
def prepare_texts(tokenizer, raw_texts: List[str], max_length: int) -> List[str]:
    """Spazi singoli e troncamento alle parole che entrano in `max_length` token."""
    from explainers import visible_text
    return [visible_text(normalize_whitespace(t), tokenizer, max_length) for t in raw_texts]


def run_signature(explainer_name: str, texts: List[str], labels: List[int], cfg) -> dict:
    """Tutto ciò che determina i risultati di una cartella: testi, parametri, seed.
    Serve a non mescolare, alla ripresa, risultati ottenuti con impostazioni diverse."""
    c = vars(cfg) if cfg is not None else {}
    relevant = {k: c.get(k) for k in ("max_length", "lime_samples", "shap_samples", "amp")}
    if explainer_name == "attention_flow":
        relevant["flow_max_length"] = c.get("flow_max_length")
    return {"explainer": explainer_name,
            "texts_sha256": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
            "labels_sha256": hashlib.sha256(",".join(map(str, labels)).encode()).hexdigest(),
            "config": relevant, "consistency_seeds": metrics.CONSISTENCY_SEEDS,
            "perturbation_ratio": metrics.PERTURBATION_RATIO,
            "perturbations_per_type": metrics.N_PERTURBATIONS_PER_TYPE,
            "explanation_seed": metrics.EXPLANATION_SEED}


def _check_signature(out_dir: Path, sig: dict, resume: bool) -> None:
    from utils import load_json
    path = out_dir / "signature.json"
    old = load_json(path)
    has_results = (out_dir / "eval.jsonl").exists() or (out_dir / "train.jsonl").exists()
    if resume and old is not None and old != sig and has_results:
        diff = sorted(k for k in set(old) | set(sig) if old.get(k) != sig.get(k))
        raise RuntimeError(f"{out_dir} contiene risultati ottenuti con impostazioni diverse ({', '.join(diff)}): "
                           "usa un'altra --out-dir oppure --no-resume")
    save_json(path, sig)


def _part_name(phase: str, shard: Optional[tuple]) -> str:
    return f"{phase}.jsonl" if shard is None else f"{phase}.s{shard[0]}of{shard[1]}.jsonl"


def _done_rows(out_dir: Path, phase: str, needed_key: Optional[str] = None,
               skip_errors: bool = False) -> Dict[int, dict]:
    """Ultima riga salvata per ogni testo, letta da tutti i file della fase (eval o
    train), qualunque sia la divisione in shard. Con skip_errors le righe con errori
    sono considerate da rifare (per esempio dopo un out-of-memory)."""
    rows = {}
    lines = []
    for path in sorted(out_dir.glob(f"{phase}*.jsonl")):
        lines.extend(read_jsonl(path))
    for r in lines:
        if needed_key is not None and needed_key not in r:
            continue
        failed = bool(r.get("error") or r.get("errors"))
        if skip_errors and failed:
            continue
        prev = rows.get(r["i"])
        if prev is not None and failed and not (prev.get("error") or prev.get("errors")):
            continue  # una riga riuscita vale più di un errore, in qualunque file sia
        rows[r["i"]] = r
    return rows


# =============================================================================
# Un explainer su un modello
# =============================================================================
def run_explainer(model, tokenizer, explainer_name: str, texts: List[str], labels: List[int],
                  metrics_to_compute: List[str], cfg, out_dir: Path, resume: bool = True,
                  shard: Optional[tuple] = None) -> dict:
    """Elabora i testi assegnati a questo processo (tutti, o quelli dello shard) e
    restituisce il riepilogo calcolato su tutto ciò che c'è nella cartella."""
    import explainers
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path, train_path = out_dir / _part_name("eval", shard), out_dir / _part_name("train", shard)
    if not resume:
        for p in (eval_path, train_path):
            p.unlink(missing_ok=True)
    mine = [i for i in range(len(texts)) if shard is None or i % shard[1] == shard[0]]

    _check_signature(out_dir, run_signature(explainer_name, texts, labels, cfg), resume)
    explain = explainers.get_explainer(explainer_name, model, tokenizer, cfg)
    mask_token = tokenizer.mask_token or tokenizer.unk_token
    timings = {}

    # ---- 1. eval: robustness + spiegazioni per la contrastivity ----
    need_eval = any(m in metrics_to_compute for m in ("robustness", "contrastivity"))
    need_rob = "robustness" in metrics_to_compute
    if need_eval:
        model.eval()
        done = _done_rows(out_dir, "eval", "robustness" if need_rob else None, skip_errors=True)
        todo = [i for i in mine if i not in done]
        print(f"  [eval] {len(mine) - len(todo)} testi già fatti, {len(todo)} da fare")
        t0, ok, consecutive_err = time.time(), 0, 0
        for n, i in enumerate(todo, 1):
            row = {"i": i, "label": labels[i]}
            try:
                set_seed(metrics.EXPLANATION_SEED)
                orig = explain(texts[i])
                row["attr"] = orig.to_dict()
                if need_rob:
                    perts = metrics.make_perturbations(texts[i], text_seed=i, mask_token=mask_token)
                    row["robustness"] = metrics.robustness_for_text(explain, texts[i], orig, perts, set_seed=set_seed)
                ok += 1
                consecutive_err = 0
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300]
                if need_rob:
                    row["robustness"] = {"robustness": None, "perturbations": []}
                consecutive_err += 1
            append_jsonl(eval_path, row)
            if ok == 0 and consecutive_err >= FAIL_FAST_AFTER:
                raise RuntimeError(f"{explainer_name}: i primi {consecutive_err} testi falliscono: {row['error']}")
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  [eval] {n}/{len(todo)}  {format_time(el)}  (stima residua {format_time(el / n * (len(todo) - n))})")
        timings["eval_seconds_this_run"] = time.time() - t0

    # ---- 2. train: consistency con dropout attivo ----
    if "consistency" in metrics_to_compute:
        done = _done_rows(out_dir, "train", skip_errors=True)
        todo = [i for i in mine if i not in done]
        print(f"  [train] {len(mine) - len(todo)} testi già fatti, {len(todo)} da fare")
        model.train()
        try:
            t0, ok, consecutive_err = time.time(), 0, 0
            for n, i in enumerate(todo, 1):
                res = metrics.consistency_for_text(explain, texts[i], metrics.CONSISTENCY_SEEDS, set_seed=set_seed)
                res["i"] = i
                if res["errors"]:
                    consecutive_err += 1
                else:
                    ok += 1
                    consecutive_err = 0
                append_jsonl(train_path, res)
                if ok == 0 and consecutive_err >= FAIL_FAST_AFTER:
                    raise RuntimeError(f"{explainer_name}: consistency fallisce: {res['errors'][0]}")
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  [train] {n}/{len(todo)}  {format_time(el)}  (stima residua {format_time(el / n * (len(todo) - n))})")
            timings["train_seconds_this_run"] = time.time() - t0
        finally:
            model.eval()

    summary = summarize(out_dir, len(texts), metrics_to_compute)
    summary["timings"] = timings
    save_json(out_dir / "summary.json", summary)
    return summary


def summarize(out_dir: Path, n_texts: int, metrics_to_compute: List[str]) -> dict:
    """Riepilogo da tutti i file della cartella (tutti gli shard)."""
    out = {"n_texts": n_texts}
    ev = {i: r for i, r in _done_rows(out_dir, "eval").items() if i < n_texts}
    out["n_eval_done"] = len(ev)
    out["n_eval_errors"] = sum(1 for r in ev.values() if "error" in r)
    if "robustness" in metrics_to_compute:
        per_text = [ev[i]["robustness"]["robustness"] if i in ev and "robustness" in ev[i] else None
                    for i in range(n_texts)]
        agg = metrics.aggregate_robustness(per_text)
        statuses = {}
        for r in ev.values():
            for p in r.get("robustness", {}).get("perturbations", []):
                key = p["status"] if not p["status"].startswith("error") else "error"
                statuses[key] = statuses.get(key, 0) + 1
        agg["perturbation_status"] = statuses
        out["robustness"] = agg
    if "contrastivity" in metrics_to_compute:
        pos = [Attribution.from_dict(r["attr"]) for r in ev.values() if "attr" in r and r["label"] == 1]
        neg = [Attribution.from_dict(r["attr"]) for r in ev.values() if "attr" in r and r["label"] == 0]
        out["contrastivity"] = metrics.contrastivity(pos, neg, exclude=MASK_STRINGS)
    if "consistency" in metrics_to_compute:
        tr = {i: r for i, r in _done_rows(out_dir, "train").items() if i < n_texts}
        out["n_train_done"] = len(tr)
        per_text = [tr[i]["consistency"] if i in tr else None for i in range(n_texts)]
        agg = metrics.aggregate_consistency(per_text)
        agg["n_invalid_pairs"] = sum(sum(1 for c in r["pair_correlations"] if c is None) for r in tr.values())
        agg["n_errors"] = sum(len(r["errors"]) for r in tr.values())
        out["consistency"] = agg
    return out


# =============================================================================
# Tabelle
# =============================================================================
def build_tables(out_root: Path, models_: List[str], explainers_: List[str]) -> Dict[str, pd.DataFrame]:
    rows = {m: {} for m in METRICS + ["contrastivity_js"]}
    for mk in models_:
        for ex in explainers_:
            s = _load_summary(out_root / mk / ex / "summary.json")
            if not s:
                continue
            if s.get("robustness"):
                rows["robustness"].setdefault(ex, {})[mk] = s["robustness"]["mean"]
            if s.get("contrastivity"):
                rows["contrastivity"].setdefault(ex, {})[mk] = s["contrastivity"]["contrastivity"]
                if s["contrastivity"].get("js_divergence") is not None:
                    rows["contrastivity_js"].setdefault(ex, {})[mk] = s["contrastivity"]["js_divergence"]
            if s.get("consistency") and s["consistency"]["mean"] is not None:
                c = s["consistency"]
                rows["consistency"].setdefault(ex, {})[mk] = f"{c['mean']:.4f}±{(c['std'] or 0):.4f}"
    tables = {}
    for m, data in rows.items():
        df = pd.DataFrame(data).T if data else pd.DataFrame()
        if not df.empty:
            df = df.reindex(index=[e for e in explainers_ if e in df.index],
                            columns=[k for k in models_ if k in df.columns])
            df.to_csv(out_root / f"{m}_table.csv")
        tables[m] = df
    return tables


def _load_summary(path: Path) -> Optional[dict]:
    from utils import load_json
    return load_json(path)


# =============================================================================
# Stima dei tempi
# =============================================================================
def estimate(models_: List[str], explainers_: List[str], metrics_to_compute: List[str], cfg,
             n_texts: int, n_probe: int = 3) -> pd.DataFrame:
    """Misura il tempo di una spiegazione su `n_probe` testi di lunghezza tipica e
    lo moltiplica per il numero di spiegazioni del run completo."""
    import explainers
    import models
    raw_texts, _ = dataset.get_clustered_sample(None)
    lengths = np.array([len(t.split()) for t in raw_texts])
    probe_idx = [int(np.argsort(np.abs(lengths - np.quantile(lengths, q)))[0])
                 for q in np.linspace(0.25, 0.75, n_probe)]
    calls = 0
    if "robustness" in metrics_to_compute:
        calls += 1 + 3 * metrics.N_PERTURBATIONS_PER_TYPE
    elif "contrastivity" in metrics_to_compute:
        calls += 1
    if "consistency" in metrics_to_compute:
        calls += len(metrics.CONSISTENCY_SEEDS)
    rows = []
    for mk in models_:
        print(f"[ESTIMATE] {mk}: caricamento...")
        model, tok = models.load_model(mk), models.load_tokenizer(mk)
        texts = prepare_texts(tok, [raw_texts[i] for i in probe_idx], cfg.max_length)
        for ex in explainers_:
            explain = explainers.get_explainer(ex, model, tok, cfg)
            set_seed(42)
            explain(texts[0])  # riscaldamento
            t0 = time.time()
            for t in texts:
                set_seed(42)
                explain(t)
            per_call = (time.time() - t0) / len(texts)
            total = per_call * calls * n_texts
            rows.append({"model": mk, "explainer": ex, "sec_per_explanation": per_call,
                         "hours_full_run": total / 3600})
            print(f"  {ex:18s} {per_call:7.3f} s/spiegazione  ->  {format_time(total)} per {n_texts} testi")
        del model
        models.clear_gpu_memory()
    df = pd.DataFrame(rows)
    print("\nOre stimate (modelli x explainer):")
    print(df.pivot(index="explainer", columns="model", values="hours_full_run").to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nTotale stimato: {df['hours_full_run'].sum():.1f} ore "
          f"({calls} spiegazioni per testo, {n_texts} testi; download dei modelli escluso)")
    return df


# =============================================================================
# Main
# =============================================================================
def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("torch", "transformers", "lime", "numpy", "scipy", "sklearn", "pandas"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:
            out[name] = None
    return out


def _parse_shard(spec: Optional[str]) -> Optional[tuple]:
    if not spec:
        return None
    i, n = (int(x) for x in spec.split("/"))
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"--shard deve essere I/N con 0 <= I < N (ricevuto {spec})")
    return (i, n) if n > 1 else None


def summarize_folder(out_root: Path, n_texts: int, metrics_to_compute: List[str]) -> Dict[str, pd.DataFrame]:
    """Ricalcola tutti i summary.json e le tabelle dai file per testo, senza caricare
    modelli (serve dopo un run diviso in più processi o in più runtime)."""
    models_ = [m for m in MODEL_ORDER if (out_root / m).is_dir()]
    models_ += sorted(p.name for p in out_root.iterdir()
                      if p.is_dir() and p.name not in MODEL_ORDER and p.name != "run_metadata")
    explainers_ = [e for e in EXPLAINERS if any((out_root / m / e).is_dir() for m in models_)]
    for mk in models_:
        for ex in explainers_:
            d = out_root / mk / ex
            if not d.is_dir():
                continue
            prev = _load_summary(d / "summary.json") or {}
            s = summarize(d, n_texts, metrics_to_compute)
            s["timings"] = prev.get("timings", {})
            save_json(d / "summary.json", s)
            print(f"  {mk}/{ex}: eval {s['n_eval_done']}/{n_texts}, train {s.get('n_train_done', '-')}/{n_texts}  {_short(s)}")
    return build_tables(out_root, models_, explainers_)


def main(argv=None):
    p = argparse.ArgumentParser(description="XAI benchmark (versione allineata alla tesi)")
    p.add_argument("--models", nargs="+", default=None, help=f"default: {MODEL_ORDER}")
    p.add_argument("--explainers", nargs="+", choices=EXPLAINERS, default=EXPLAINERS)
    p.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)
    p.add_argument("--sample", type=int, default=400, help="testi (default 400, tutto il campione)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--flow-max-length", type=int, default=128)
    p.add_argument("--flow-workers", type=int, default=1)
    p.add_argument("--lime-samples", type=int, default=500)
    p.add_argument("--shap-samples", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-amp", action="store_true", help="niente float16 nei forward di LIME/SHAP")
    p.add_argument("--device", default=None, help="cuda o cpu (default: cuda se disponibile)")
    p.add_argument("--threads", type=int, default=None, help="thread di PyTorch per questo processo")
    p.add_argument("--shard", default=None, metavar="I/N", help="elabora solo i testi con indice %% N == I")
    p.add_argument("--out-dir", default="runs/v2")
    p.add_argument("--no-resume", action="store_true",
                   help="ricomincia da zero (cancella i .jsonl di questo processo/shard)")
    p.add_argument("--local-model", nargs="*", default=[], metavar="KEY=PATH",
                   help="modelli locali aggiuntivi (usati dai test)")
    p.add_argument("--estimate", action="store_true", help="stima i tempi e termina")
    p.add_argument("--summarize-only", action="store_true",
                   help="ricalcola riepiloghi e tabelle dai file esistenti, senza caricare modelli")
    args = p.parse_args(argv)
    shard = _parse_shard(args.shard)
    out_root = Path(args.out_dir)

    if args.summarize_only:
        n_texts = len(dataset.get_clustered_sample(args.sample)[0])
        tables = summarize_folder(out_root, n_texts, args.metrics)
        _print_tables(tables)
        return 0

    if args.threads:
        try:
            import torch
            torch.set_num_threads(args.threads)
        except ImportError:
            pass
    import explainers
    import models
    if args.device:
        models.set_device(args.device)
    for spec in args.local_model:
        k, path = spec.split("=", 1)
        models.register_model(k, path)
    models_ = args.models or [m for m in MODEL_ORDER if m in models.MODELS]
    unknown = [m for m in models_ if m not in models.MODELS]
    if unknown:
        p.error(f"modelli sconosciuti: {unknown}")

    cfg = explainers.ExplainerConfig(max_length=args.max_length, flow_max_length=args.flow_max_length,
                                     flow_workers=args.flow_workers, lime_samples=args.lime_samples,
                                     shap_samples=args.shap_samples, batch_size=args.batch_size,
                                     amp=not args.no_amp)
    raw_texts, labels = dataset.get_clustered_sample(args.sample)
    tag = "" if shard is None else f" shard {shard[0]}/{shard[1]}"
    print(f"[DATA] {len(raw_texts)} testi ({sum(labels)} positivi), device {models.DEVICE}{tag}")

    if args.estimate:
        estimate(models_, args.explainers, args.metrics, cfg, n_texts=len(raw_texts))
        return 0

    out_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
    meta = {"run_id": run_id, "git_commit": git_commit(), "argv": sys.argv, "config": vars(cfg),
            "shard": shard, "metrics": args.metrics, "explainers": args.explainers,
            "models": {m: models.MODELS[m] for m in models_},
            "n_texts": len(raw_texts), "n_positive": int(sum(labels)),
            "dataset_sha256": hashlib.sha256("\n".join(raw_texts).encode()).hexdigest(),
            "consistency_seeds": metrics.CONSISTENCY_SEEDS, "perturbation_ratio": metrics.PERTURBATION_RATIO,
            "perturbations_per_type": metrics.N_PERTURBATIONS_PER_TYPE,
            "contrastivity_vocab_size": metrics.CONTRASTIVITY_VOCAB_SIZE,
            "device": str(models.DEVICE), "versions": library_versions(), "truncation": {}}
    meta_path = out_root / "run_metadata" / f"{run_id}.json"
    shard_tag = "" if shard is None else f".s{shard[0]}of{shard[1]}"

    for mk in models_:
        print(f"\n{'=' * 70}\n MODELLO {mk}{tag}\n{'=' * 70}")
        with Timer(f"caricamento {mk}"):
            model, tok = models.load_model(mk), models.load_tokenizer(mk)
        texts = prepare_texts(tok, raw_texts, cfg.max_length)
        n_cut = sum(len(t.split(" ")) < len(normalize_whitespace(r).split(" ")) for t, r in zip(texts, raw_texts))
        meta["truncation"][mk] = {"texts_truncated_at_max_length": int(n_cut)}
        print(f"[DATA] {n_cut}/{len(texts)} testi troncati a {cfg.max_length} token")
        for ex in args.explainers:
            print(f"\n[{mk}] {ex}{tag}")
            try:
                with Timer(f"{mk}/{ex}"):
                    s = run_explainer(model, tok, ex, texts, labels, args.metrics, cfg,
                                      out_root / mk / ex, resume=not args.no_resume, shard=shard)
                print(f"  -> {_short(s)}")
                (out_root / mk / ex / f"failed{shard_tag}.json").unlink(missing_ok=True)
            except Exception as e:
                print(f"  [ERRORE] {mk}/{ex}: {type(e).__name__}: {e}")
                save_json(out_root / mk / ex / f"failed{shard_tag}.json", {"error": f"{type(e).__name__}: {e}"})
        del model
        models.clear_gpu_memory()
        save_json(meta_path, meta)

    save_json(meta_path, meta)
    if shard is None:
        _print_tables(build_tables(out_root, models_, args.explainers))
    else:
        print("\n[SHARD] fatto. Quando tutti i processi hanno finito: python report.py --summarize-only "
              f"--out-dir {out_root}")
    return 0


def _print_tables(tables: Dict[str, pd.DataFrame]) -> None:
    for m, df in tables.items():
        if not df.empty:
            print(f"\n{m.upper()}\n{df.to_string(float_format=lambda x: f'{x:.4f}')}")


def _short(s: dict) -> str:
    parts = []
    if s.get("robustness"):
        r = s["robustness"]
        parts.append(f"robustness {r['mean']:.4f} (n={r['n_valid']})" if r["mean"] is not None else "robustness n/d")
    if s.get("consistency"):
        c = s["consistency"]
        parts.append(f"consistency {c['mean']:.4f}±{(c['std'] or 0):.4f} (n={c['n_valid']})" if c["mean"] is not None else "consistency n/d")
    if s.get("contrastivity"):
        v = s["contrastivity"]["contrastivity"]
        parts.append(f"contrastivity {v:.4f}" if v is not None else "contrastivity n/d")
    return ", ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
