"""
parallel_run.py – Tutto il benchmark con il massimo parallelismo dell'hardware disponibile
=========================================================================================

    python parallel_run.py --out-dir /content/drive/MyDrive/xai_v2

Divide il lavoro in processi indipendenti di report.py e li fa girare insieme:

- corsie GPU: LIME, SHAP, Input×Gradient, Rollout e LRP, con i testi divisi in shard;
- processi CPU: Attention Flow (il flusso massimo usa solo la CPU), uno per core libero,
  con più shard per i modelli più profondi (il costo cresce con il numero di layer).

Ogni testo ha i propri seed, quindi i risultati sono identici a quelli di un'unica
esecuzione sequenziale: cambia solo il tempo. Si interrompe e si rilancia con lo
stesso comando: le parti già fatte vengono saltate. Alla fine ricalcola riepiloghi e
tabelle (report.py --summarize-only).

Per usare due runtime nello stesso momento (per esempio uno GPU e uno solo CPU) sulla
stessa cartella di Google Drive:

    python parallel_run.py --role gpu  --out-dir ...   # runtime GPU
    python parallel_run.py --role flow --out-dir ...   # runtime CPU

Le opzioni non riconosciute (per esempio --lime-samples, --sample) passano a report.py.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
NON_FLOW = ["lime", "shap", "grad_input", "attention_rollout", "lrp"]
MODEL_ORDER = ["tinybert", "distilbert", "roberta-base", "bert-large", "roberta-large"]
# costo relativo per modello: parametri (corsie GPU) e numero di layer (Attention Flow)
GPU_COST = {"tinybert": 14, "distilbert": 66, "roberta-base": 125, "bert-large": 335, "roberta-large": 355}
FLOW_COST = {"tinybert": 4, "distilbert": 6, "roberta-base": 12, "bert-large": 24, "roberta-large": 24}


class Job:
    def __init__(self, name: str, models: List[str], explainers: List[str], shard: tuple, device: str,
                 cost: float, threads: int):
        self.name, self.models, self.explainers, self.shard = name, models, explainers, shard
        self.device, self.cost, self.threads = device, cost, threads
        self.proc: Optional[subprocess.Popen] = None
        self.log = None
        self.started = self.ended = None

    def args(self, out_dir: Path, extra: List[str]) -> List[str]:
        a = [sys.executable, str(HERE / "report.py"), "--out-dir", str(out_dir), "--models", *self.models,
             "--explainers", *self.explainers, "--device", self.device, "--threads", str(self.threads)]
        if self.shard[1] > 1:
            a += ["--shard", f"{self.shard[0]}/{self.shard[1]}"]
        return a + extra


def split_units(models: List[str], cost: Dict[str, float], slots: int) -> Dict[str, int]:
    """Numero di shard per modello, proporzionale al costo (circa due unità per slot)."""
    total = sum(cost.get(m, 1) for m in models)
    return {m: max(1, round(2 * slots * cost.get(m, 1) / total)) for m in models}


def plan(models: List[str], role: str, has_cuda: bool, cpus: int, gpu_lanes: Optional[int],
         flow_procs: Optional[int]) -> Dict[str, list]:
    pools = {}
    want_gpu = role in ("all", "gpu")
    want_flow = role in ("all", "flow")
    if want_gpu:
        if has_cuda:
            lanes = gpu_lanes or (1 if cpus <= 2 else 2)
            dev, threads = "cuda", 1
        else:
            lanes = gpu_lanes or (max(1, cpus // 2) if want_flow else cpus)
            dev, threads = "cpu", 1
        jobs = []
        for m, k in split_units(models, GPU_COST, lanes).items():
            for i in range(k):
                jobs.append(Job(f"{m}-metodi-s{i}of{k}", [m], NON_FLOW, (i, k), dev,
                                GPU_COST.get(m, 1) / k, threads))
        pools["gpu" if has_cuda else "cpu-metodi"] = [lanes, jobs]
    if want_flow:
        used = pools.get("gpu", pools.get("cpu-metodi", [0]))[0] if want_gpu else 0
        procs = flow_procs or max(1, cpus - used)
        jobs = []
        for m, k in split_units(models, FLOW_COST, procs).items():
            for i in range(k):
                jobs.append(Job(f"{m}-flow-s{i}of{k}", [m], ["attention_flow"], (i, k), "cpu",
                                FLOW_COST.get(m, 1) / k, 1))
        pools["flow"] = [procs, jobs]
    for p in pools.values():
        p[1].sort(key=lambda j: -j.cost)  # prima i lavori più lunghi
    return pools


# ---------------------------------------------------------------- avanzamento
def _count_done(out_dir: Path, model: str, explainer: str, shard: tuple, n_texts: int) -> int:
    sys.path.insert(0, str(HERE))
    from report import _done_rows
    d = out_dir / model / explainer
    if not d.is_dir():
        return 0
    mine = {i for i in range(n_texts) if i % shard[1] == shard[0]}
    ev = _done_rows(d, "eval", "robustness", skip_errors=True)
    tr = _done_rows(d, "train", skip_errors=True)
    return sum(1 for i in mine if i in ev) + sum(1 for i in mine if i in tr)


def job_done(job: Job, out_dir: Path, n_texts: int) -> bool:
    need = 2 * len([i for i in range(n_texts) if i % job.shard[1] == job.shard[0]])
    return all(_count_done(out_dir, m, e, job.shard, n_texts) >= need for m in job.models for e in job.explainers)


def progress(out_dir: Path, models: List[str], explainers: List[str], n_texts: int) -> str:
    sys.path.insert(0, str(HERE))
    from report import _done_rows
    parts = []
    tot_done = tot = 0
    for e in explainers:
        done = 0
        for m in models:
            d = out_dir / m / e
            if d.is_dir():
                done += len(_done_rows(d, "eval", "robustness", skip_errors=True))
                done += len(_done_rows(d, "train", skip_errors=True))
        need = 2 * n_texts * len(models)
        parts.append(f"{e} {100 * done / need:.0f}%")
        tot_done += done
        tot += need
    return f"totale {100 * tot_done / max(tot, 1):.1f}% | " + ", ".join(parts)


# ---------------------------------------------------------------- esecuzione
def run_pools(pools: Dict[str, list], out_dir: Path, extra: List[str], n_texts: int, models: List[str],
              explainers: List[str], poll: int) -> List[Job]:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    queues = {name: list(jobs) for name, (slots, jobs) in pools.items()}
    running = {name: [] for name in pools}
    finished, failed = [], []
    t0 = last = time.time()
    while any(queues.values()) or any(running.values()):
        for name, (slots, _) in pools.items():
            for j in list(running[name]):
                if j.proc.poll() is not None:
                    j.ended = time.time()
                    j.log.close()
                    running[name].remove(j)
                    (finished if j.proc.returncode == 0 else failed).append(j)
                    status = "ok" if j.proc.returncode == 0 else f"ERRORE (codice {j.proc.returncode})"
                    print(f"[{time.strftime('%H:%M')}] fine {j.name}: {status}, {(j.ended - j.started) / 60:.0f} min", flush=True)
            while queues[name] and len(running[name]) < slots:
                j = queues[name].pop(0)
                if job_done(j, out_dir, n_texts):
                    print(f"[{time.strftime('%H:%M')}] già fatto: {j.name}", flush=True)
                    finished.append(j)
                    continue
                j.log = open(logs / f"{j.name}.log", "a", encoding="utf-8")
                j.started = time.time()
                j.proc = subprocess.Popen(j.args(out_dir, extra), cwd=HERE, env=env,
                                          stdout=j.log, stderr=subprocess.STDOUT)
                running[name].append(j)
                print(f"[{time.strftime('%H:%M')}] avvio {j.name} ({name})", flush=True)
        if time.time() - last >= poll:
            last = time.time()
            busy = ", ".join(j.name for js in running.values() for j in js)
            print(f"[{time.strftime('%H:%M')}] {(last - t0) / 3600:.1f} h | "
                  f"{progress(out_dir, models, explainers, n_texts)} | in corso: {busy}", flush=True)
        time.sleep(2)
    return failed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models", nargs="+", default=MODEL_ORDER)
    p.add_argument("--role", choices=["all", "gpu", "flow"], default="all",
                   help="gpu: solo i cinque metodi senza Flow; flow: solo Attention Flow")
    p.add_argument("--gpu-lanes", type=int, default=None, help="processi sulla GPU (default 2, 1 con <= 2 CPU)")
    p.add_argument("--flow-procs", type=int, default=None, help="processi per Attention Flow (default: CPU libere)")
    p.add_argument("--sample", type=int, default=400)
    p.add_argument("--poll", type=int, default=300, help="secondi tra due righe di avanzamento")
    p.add_argument("--dry-run", action="store_true", help="mostra il piano e termina")
    args, extra = p.parse_known_args(argv)
    extra = ["--sample", str(args.sample)] + extra

    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    cpus = os.cpu_count() or 1
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = plan(args.models, args.role, has_cuda, cpus, args.gpu_lanes, args.flow_procs)

    print(f"[PIANO] {cpus} CPU, GPU {'sì' if has_cuda else 'no'}, ruolo {args.role}")
    for name, (slots, jobs) in pools.items():
        print(f"  {name}: {slots} processi in parallelo, {len(jobs)} lavori: " + ", ".join(j.name for j in jobs))
    if args.dry_run:
        return 0

    explainers = (NON_FLOW if args.role in ("all", "gpu") else []) + (["attention_flow"] if args.role in ("all", "flow") else [])
    failed = run_pools(pools, out_dir, extra, args.sample, args.models, explainers, args.poll)

    print("\n[RIEPILOGO] tabelle da tutti i file della cartella")
    subprocess.run([sys.executable, str(HERE / "report.py"), "--summarize-only", "--out-dir", str(out_dir),
                    "--sample", str(args.sample)], cwd=HERE, check=False)
    if failed:
        print("\n[ATTENZIONE] lavori non riusciti (rilancia lo stesso comando per riprenderli):")
        for j in failed:
            tail = (out_dir / "logs" / f"{j.name}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-5:]
            print(f"  {j.name}:\n    " + "\n    ".join(tail))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
