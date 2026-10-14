"""
metrics.py – Robustness, Consistency e Contrastivity come definite nella tesi (cap. 3.1)
======================================================================================

Il modulo non dipende da PyTorch: le metriche ricevono l'explainer come funzione
`testo -> Attribution` e la funzione che fissa i seed, quindi si possono verificare
anche con explainer finti.

Robustness (pp. 45-47, eq. 3.1-3.3)
    Per ogni testo x e ogni perturbazione x' in P: dissimilarità media |s_t(x) - s'_t(x')|
    sulle parole presenti in entrambe le spiegazioni, con punteggi normalizzati in
    [0, 1]. Robustness(x) = media sulle perturbazioni; poi media sui testi.
    Più basso = più stabile.

Consistency (pp. 47-49, eq. 3.4-3.6)
    Stesso testo, dropout attivo, quattro seed (42, 123, 456, 789): media delle sei
    correlazioni di Spearman tra coppie di spiegazioni; poi media e deviazione
    standard (N-1) sui testi.

Contrastivity (pp. 49-50)
    Somma dei punteggi per parola nelle spiegazioni dei testi positivi e negativi,
    normalizzazione a distribuzione (sottraendo il minimo), vocabolario ristretto
    alle 1.000 parole più frequenti, D_KL(P+ || P-) in base 2 con eps = 1e-10.
    La tesi non dice su quale scala sommare i punteggi. Qui ogni spiegazione è
    divisa per il suo massimo valore assoluto: il segno resta (serve al passo
    "S - min S") e ogni testo pesa allo stesso modo. Con i punteggi in [0, 1] della
    robustness tutte le parole assenti da una classe avrebbero S = 0 = min S, quindi
    probabilità 0, e la divergenza sarebbe dominata da eps. Per come è scritta la
    formula, la parola con S minimo ha comunque probabilità 0: il conteggio è in
    `n_vocab_zero_in_neg`.

Valori non calcolabili (per esempio una spiegazione costante, per cui Spearman non è
definita) sono NaN: vengono esclusi dalle medie e contati, non sostituiti con
valori arbitrari.
"""

import math
import random
import re
import warnings
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from attribution import Attribution

# ==== Parametri della tesi ====
PERTURBATION_RATIO = 0.15                       # rho, p. 47
N_PERTURBATIONS_PER_TYPE = 2                    # perturbazioni per strategia (|P| = 6)
CONSISTENCY_SEEDS = [42, 123, 456, 789]         # p. 48
CONTRASTIVITY_VOCAB_SIZE = 1000                 # p. 50
KL_EPSILON = 1e-10                              # p. 50
EXPLANATION_SEED = 42                           # seed fisso per le spiegazioni senza dropout

# ==== Dizionario di sinonimi (Random Substitution, p. 46) ====
# Sostituzioni di una parola con un'altra di significato e polarità equivalenti.
# Il dizionario della versione della tesi aveva 12 voci, e in gran parte dei testi
# non trovava nulla da sostituire: qui è esteso a parole frequenti nelle recensioni.
SYNONYMS: Dict[str, str] = {
    # cinema
    "movie": "film", "film": "movie", "movies": "films", "films": "movies",
    "actor": "performer", "actors": "performers", "actress": "performer",
    "director": "filmmaker", "story": "tale", "plot": "storyline", "storyline": "plot",
    "scene": "sequence", "scenes": "sequences", "character": "role", "characters": "roles",
    "cast": "ensemble", "ending": "finale", "audience": "viewers", "viewers": "audience",
    "picture": "film", "flick": "movie", "series": "show",
    "role": "part", "roles": "parts", "screenplay": "script", "script": "screenplay",
    "watch": "view", "watched": "viewed", "watching": "viewing", "saw": "watched",
    "seen": "watched", "performance": "portrayal", "performances": "portrayals",
    # aggettivi positivi
    "good": "fine", "great": "terrific", "nice": "pleasant", "excellent": "outstanding",
    "amazing": "astonishing", "wonderful": "marvelous", "fantastic": "fabulous",
    "beautiful": "lovely", "brilliant": "superb", "superb": "brilliant",
    "enjoyable": "entertaining", "entertaining": "enjoyable", "funny": "hilarious",
    "interesting": "intriguing", "best": "finest", "favorite": "preferred",
    "perfect": "flawless", "loved": "adored", "love": "adore",     "liked": "enjoyed", "enjoyed": "liked", "recommend": "suggest",     # aggettivi negativi
    "bad": "poor", "poor": "weak", "terrible": "awful", "awful": "dreadful",
    "horrible": "terrible", "boring": "dull", "dull": "tedious", "worst": "poorest",
    "stupid": "dumb", "dumb": "stupid", "waste": "squander", "hate": "loathe",
    "hated": "loathed", "annoying": "irritating", "ridiculous": "absurd",
    "silly": "foolish", "weak": "feeble", "mess": "shambles", "disappointing": "underwhelming",
    "pointless": "meaningless", "lame": "feeble",
    # parole frequenti neutre
    "very": "really", "really": "truly", "quite": "rather",     "lot": "great deal", "many": "numerous", "big": "large", "small": "little",
    "little": "small",     "begin": "start", "people": "folks",
    "man": "guy", "guy": "man", "woman": "lady", "kids": "children", "children": "kids",
    "old": "aged", "new": "recent", "real": "genuine", "whole": "entire", "entire": "whole",
    "think": "believe", "thought": "believed", "know": "realize", "seems": "appears",
    "seemed": "appeared", "made": "created",     "find": "discover", "found": "discovered",
    "also": "too", "almost": "nearly", "probably": "likely", "actually": "really",
    "completely": "totally", "totally": "completely", "especially": "particularly",
    "simply": "just", "just": "simply", "maybe": "perhaps", "however": "nevertheless",
    "because": "since", "although": "though", "though": "although",     "fact": "truth", "part": "portion",
    "minutes": "mins", "world": "globe",
    "life": "existence", "family": "household", "friends": "pals", "house": "home",
    "home": "house", "town": "city", "city": "town", "car": "vehicle",
}

_PUNCT_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def _substitute_word(word: str) -> Optional[str]:
    """Sinonimo di `word` mantenendo punteggiatura e maiuscola iniziale, o None."""
    m = _PUNCT_RE.match(word)
    pre, core, post = m.group(1), m.group(2), m.group(3)
    syn = SYNONYMS.get(core.lower())
    if syn is None or not core:
        return None
    if core[0].isupper():
        syn = syn[0].upper() + syn[1:]
    return pre + syn + post


# ==== Perturbazioni (p. 46) ====
# Ognuna restituisce (parole perturbate, coppie (indice originale, indice perturbato)
# delle parole rimaste identiche), oppure None se la perturbazione non è applicabile.
Perturbation = Tuple[List[str], List[Tuple[int, int]]]


def _n_to_change(n_words: int, ratio: float) -> int:
    # floor(rho * |x|) come nella tesi, con almeno una parola cambiata e almeno una intatta
    return min(max(1, math.floor(ratio * n_words)), n_words - 1)


def perturb_mask(words: List[str], rng: random.Random, ratio: float = PERTURBATION_RATIO,
                 mask_token: str = "[MASK]") -> Optional[Perturbation]:
    """Random Masking: floor(rho·|x|) parole casuali sostituite con il token di mask
    del tokenizer ([MASK] per BERT, <mask> per RoBERTa)."""
    n = len(words)
    if n < 2:
        return None
    idx = set(rng.sample(range(n), _n_to_change(n, ratio)))
    new = [mask_token if i in idx else w for i, w in enumerate(words)]
    pairs = [(i, i) for i in range(n) if i not in idx]
    return new, pairs


def perturb_delete(words: List[str], rng: random.Random, ratio: float = PERTURBATION_RATIO,
                   **_) -> Optional[Perturbation]:
    """Random Deletion: floor(rho·|x|) parole casuali eliminate."""
    n = len(words)
    if n < 2:
        return None
    drop = set(rng.sample(range(n), _n_to_change(n, ratio)))
    kept = [i for i in range(n) if i not in drop]
    return [words[i] for i in kept], [(i, j) for j, i in enumerate(kept)]


def perturb_substitute(words: List[str], rng: random.Random, ratio: float = PERTURBATION_RATIO,
                       **_) -> Optional[Perturbation]:
    """Random Substitution: fino a floor(rho·|x|) parole sostituite con un sinonimo
    del dizionario, scelte a caso tra quelle che ne hanno uno. Una sostituzione può
    produrre due parole ("lot" -> "great deal"): le coppie tengono conto dello spostamento."""
    n = len(words)
    cand = [i for i, w in enumerate(words) if _substitute_word(w) is not None]
    if n < 2 or not cand:
        return None
    k = min(_n_to_change(n, ratio), len(cand))
    chosen = set(rng.sample(cand, k))
    new, pairs = [], []
    for i, w in enumerate(words):
        if i in chosen:
            new.extend(_substitute_word(w).split(" "))
        else:
            pairs.append((i, len(new)))
            new.append(w)
    return new, pairs


PERTURBATIONS = {"mask": perturb_mask, "delete": perturb_delete, "substitute": perturb_substitute}


def make_perturbations(text: str, text_seed: int, mask_token: str,
                       n_per_type: int = N_PERTURBATIONS_PER_TYPE,
                       ratio: float = PERTURBATION_RATIO) -> List[dict]:
    """Insieme P delle perturbazioni di un testo, riproducibile dato `text_seed`."""
    words = text.split(" ")
    out = []
    for name, fn in PERTURBATIONS.items():
        for k in range(n_per_type):
            rng = random.Random(f"{text_seed}-{name}-{k}")
            res = fn(words, rng, ratio=ratio, mask_token=mask_token)
            if res is None:
                out.append({"type": name, "k": k, "text": None, "pairs": []})
            else:
                new, pairs = res
                out.append({"type": name, "k": k, "text": " ".join(new), "pairs": pairs})
    return out


# ==== Seed ====
def default_set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ==== 1. Robustness ====
def dissimilarity(orig: Attribution, pert: Attribution, pairs: Sequence[Tuple[int, int]]) -> float:
    """Eq. 3.2: media di |s_t(x) - s'_t(x')| sulle parole condivise (visibili in entrambe)."""
    a, b = orig.scores, pert.scores
    diffs = [abs(a[i] - b[j]) for i, j in pairs if i < len(a) and j < len(b)]
    return float(np.mean(diffs)) if diffs else float("nan")


def robustness_for_text(explain: Callable[[str], Attribution], text: str, orig: Attribution,
                        perturbations: List[dict], set_seed: Callable[[int], None] = default_set_seed,
                        seed: int = EXPLANATION_SEED) -> dict:
    """Eq. 3.1 per un testo. `orig` è la spiegazione del testo originale."""
    per_pert = []
    for p in perturbations:
        if p["text"] is None:
            per_pert.append({"type": p["type"], "value": None, "status": "not_applicable"})
            continue
        try:
            set_seed(seed)
            pert_attr = explain(p["text"])
            d = dissimilarity(orig, pert_attr, p["pairs"])
            status = "ok" if np.isfinite(d) else "no_shared_words"
        except Exception as e:  # registrato, non trasformato in un punteggio
            d, status = float("nan"), f"error: {type(e).__name__}: {e}"[:200]
        per_pert.append({"type": p["type"], "value": None if not np.isfinite(d) else d, "status": status})
    vals = [x["value"] for x in per_pert if x["value"] is not None]
    return {"robustness": float(np.mean(vals)) if vals else None, "perturbations": per_pert}


def aggregate_robustness(per_text: Sequence[Optional[float]]) -> dict:
    """Eq. 3.3: media sui testi con un valore definito."""
    vals = np.array([v for v in per_text if v is not None], dtype=float)
    return {"mean": float(vals.mean()) if len(vals) else None,
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "n_valid": int(len(vals)), "n_total": int(len(per_text))}


# ==== 2. Consistency ====
def spearman_same_text(a: Attribution, b: Attribution) -> float:
    """Spearman tra due spiegazioni dello stesso testo, parola per parola (stessa posizione).

    NaN se le spiegazioni hanno lunghezze diverse o se una delle due è costante."""
    if len(a.scores) != len(b.scores) or len(a.scores) < 2:
        return float("nan")
    x, y = np.asarray(a.raw_scores, float), np.asarray(b.raw_scores, float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(x, y)[0]
    return float(rho)


def consistency_for_text(explain: Callable[[str], Attribution], text: str,
                         seeds: Sequence[int] = CONSISTENCY_SEEDS,
                         set_seed: Callable[[int], None] = default_set_seed) -> dict:
    """Eq. 3.4 per un testo: media delle correlazioni tra le coppie di seed.

    Il dropout va acceso dal chiamante (model.train()) prima di chiamare questa funzione."""
    attrs, errors = {}, []
    for s in seeds:
        try:
            set_seed(s)
            attrs[s] = explain(text)
        except Exception as e:
            errors.append(f"seed {s}: {type(e).__name__}: {e}"[:200])
    corrs = []
    for i, sa in enumerate(seeds):
        for sb in seeds[i + 1:]:
            if sa in attrs and sb in attrs:
                corrs.append(spearman_same_text(attrs[sa], attrs[sb]))
    valid = [c for c in corrs if np.isfinite(c)]
    return {"consistency": float(np.mean(valid)) if valid else None,
            "pair_correlations": [None if not np.isfinite(c) else c for c in corrs],
            "n_valid_pairs": len(valid), "errors": errors}


def aggregate_consistency(per_text: Sequence[Optional[float]]) -> dict:
    """Eq. 3.5-3.6: media e deviazione standard campionaria (N-1) sui testi."""
    vals = np.array([v for v in per_text if v is not None], dtype=float)
    return {"mean": float(vals.mean()) if len(vals) else None,
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "n_valid": int(len(vals)), "n_total": int(len(per_text))}


# ==== 3. Contrastivity ====
_STRIP_RE = re.compile(r"^\W+|\W+$", re.UNICODE)


def normalize_word(word: str) -> str:
    """Forma della parola usata per il vocabolario: minuscolo, senza punteggiatura ai bordi."""
    return _STRIP_RE.sub("", word.lower())


def contrastivity(pos_attrs: Sequence[Attribution], neg_attrs: Sequence[Attribution],
                  vocab_size: int = CONTRASTIVITY_VOCAB_SIZE, eps: float = KL_EPSILON,
                  exclude: Sequence[str] = (), scale: str = "signed") -> dict:
    """D_KL(P+ || P-) in base 2 (p. 50).

    1. S_c(t) = somma dei punteggi della parola t nelle spiegazioni della classe c;
    2. P_c(t) = (S_c(t) - min S_c) / somma(S_c - min S_c), sul vocabolario V;
    3. V = le `vocab_size` parole più frequenti (occorrenze in tutte le spiegazioni,
       a parità di frequenza in ordine alfabetico).

    scale: "signed" (default, punteggi / max|punteggio|), "raw" o "minmax01"."""
    pick = {"signed": lambda a: a.signed_scores, "raw": lambda a: a.raw_scores,
            "minmax01": lambda a: a.scores}[scale]
    exclude = {normalize_word(w) for w in exclude}
    S = {"pos": Counter(), "neg": Counter()}
    freq = Counter()
    for cls, attrs in (("pos", pos_attrs), ("neg", neg_attrs)):
        for a in attrs:
            for w, s in zip(a.words, pick(a)):
                t = normalize_word(w)
                if not t or t in exclude:
                    continue
                S[cls][t] += float(s)
                freq[t] += 1
    if len(freq) < 2 or not pos_attrs or not neg_attrs:
        return {"contrastivity": None, "vocab_size": len(freq)}
    vocab = [t for t, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:vocab_size]]

    def dist(counter):
        v = np.array([counter.get(t, 0.0) for t in vocab], dtype=float)
        v = v - v.min()
        tot = v.sum()
        return v / tot if tot > 0 else np.full(len(v), 1.0 / len(v))

    p, q = dist(S["pos"]), dist(S["neg"])
    mask = p > 0
    kl = float(np.sum(p[mask] * np.log2(p[mask] / (q[mask] + eps))))
    return {"contrastivity": kl, "vocab_size": len(vocab), "n_pos": len(pos_attrs), "n_neg": len(neg_attrs),
            "scale": scale, "n_vocab_zero_in_neg": int(np.sum(mask & (q == 0)))}
