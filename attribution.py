"""
attribution.py – Strutture e calcoli comuni agli explainer (solo numpy/scipy)
============================================================================

Qui sta tutto ciò che non richiede PyTorch, così può essere verificato a parte:

- `Attribution`: una spiegazione a livello di parola (parole separate da spazi);
- aggregazione dei punteggi dai token del tokenizer alle parole;
- normalizzazione min-max in [0, 1] (tesi, p. 46);
- Attention Rollout (tesi, eq. 2.20-2.24) e Attention Flow (tesi, pp. 39-40).

Unità di spiegazione. LIME e SHAP lavorano naturalmente sulle parole, i metodi
basati su gradiente e attenzione sui token del tokenizer (sotto-parole). Per
confrontare i sei metodi sulla stessa unità, ogni spiegazione è riportata alla
parola: il punteggio di una parola è la somma dei punteggi dei suoi token.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


# ==== Spiegazione a livello di parola ====
@dataclass
class Attribution:
    """Spiegazione di un testo: un punteggio per ciascuna parola visibile al modello.

    words       parole del testo (split su spazi) effettivamente viste dal modello,
                nell'ordine del testo; le parole oltre il troncamento sono escluse
    raw_scores  punteggi originali del metodo (segno incluso)
    scores      punteggi normalizzati min-max in [0, 1]
    n_words_total  numero di parole del testo prima del troncamento
    """
    words: List[str]
    raw_scores: List[float]
    scores: List[float] = field(default_factory=list)
    n_words_total: int = 0

    def __post_init__(self):
        if len(self.words) != len(self.raw_scores):
            raise ValueError("words e raw_scores devono avere la stessa lunghezza")
        if not self.scores:
            self.scores = minmax01(self.raw_scores).tolist()
        if not self.n_words_total:
            self.n_words_total = len(self.words)

    @property
    def signed_scores(self) -> List[float]:
        """Punteggi con segno divisi per il massimo valore assoluto della spiegazione,
        in [-1, 1] (usati dalla contrastivity, vedi metrics.contrastivity)."""
        arr = np.asarray(self.raw_scores, dtype=float)
        m = np.max(np.abs(arr)) if arr.size else 0.0
        return (arr / m if m > 0 else np.zeros_like(arr)).tolist()

    # compatibilità con il codice precedente, che usava .tokens
    @property
    def tokens(self) -> List[str]:
        return self.words

    def to_dict(self) -> dict:
        return {"words": self.words, "raw_scores": [float(x) for x in self.raw_scores],
                "scores": [float(x) for x in self.scores], "n_words_total": self.n_words_total}

    @classmethod
    def from_dict(cls, d: dict) -> "Attribution":
        return cls(d["words"], d["raw_scores"], d["scores"], d.get("n_words_total", 0))

    def __repr__(self):
        items = ", ".join(f"{w}:{s:.3f}" for w, s in zip(self.words[:3], self.scores[:3]))
        return f"Attribution({len(self.words)} parole: {items}...)"


def minmax01(scores: Sequence[float]) -> np.ndarray:
    """Normalizzazione min-max in [0, 1]. Un vettore costante diventa tutto 0."""
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = np.min(arr), np.max(arr)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("punteggi non finiti")
    if hi - lo <= 0:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ==== Dai token del tokenizer alle parole ====
def normalize_whitespace(text: str) -> str:
    """Riduce ogni sequenza di spazi/a capo a uno spazio singolo.

    Tutti gli explainer e le perturbazioni lavorano su `text.split(" ")`, quindi il
    testo deve avere un solo spazio tra parole consecutive."""
    return " ".join(text.split())


def word_spans(text: str) -> List[tuple]:
    """(inizio, fine) in caratteri di ogni parola di un testo a spazi singoli."""
    spans, pos = [], 0
    for w in text.split(" "):
        if w:
            spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    return spans


def token_word_ids(offsets: Sequence[Sequence[int]], spans: Sequence[tuple]) -> List[int]:
    """Indice della parola di ogni token, dato l'offset (inizio, fine) in caratteri.

    I token speciali ([CLS], [SEP], <s>, ...) hanno offset (0, 0) e ricevono -1.
    Un token appartiene alla parola che contiene il suo primo carattere (gli spazi
    iniziali di alcuni tokenizer BPE sono ignorati)."""
    starts = np.array([s for s, _ in spans]) if spans else np.array([], dtype=int)
    ends = np.array([e for _, e in spans]) if spans else np.array([], dtype=int)
    out = []
    for s, e in offsets:
        s, e = int(s), int(e)
        if e <= s:
            out.append(-1)
            continue
        idx = np.searchsorted(starts, s, side="right") - 1
        if idx < 0:
            # il token inizia nello spazio prima della prima parola
            idx = 0 if len(starts) else -1
        elif s >= ends[idx]:
            # il token inizia nello spazio tra due parole: appartiene alla successiva
            idx = idx + 1 if idx + 1 < len(starts) else -1
        out.append(int(idx))
    return out


def aggregate_to_words(token_scores: Sequence[float], word_ids: Sequence[int]) -> np.ndarray:
    """Somma i punteggi dei token di ogni parola. Restituisce un vettore lungo quanto
    il numero di parole visibili (ultima parola con almeno un token + 1)."""
    word_ids = np.asarray(word_ids)
    valid = word_ids >= 0
    if not valid.any():
        return np.zeros(0)
    n = int(word_ids[valid].max()) + 1
    out = np.zeros(n)
    np.add.at(out, word_ids[valid], np.asarray(token_scores, dtype=float)[valid])
    return out


# ==== Attention Rollout e Attention Flow ====
def residual_attention(att: np.ndarray) -> np.ndarray:
    """A(l) = 1/2 W(l) + 1/2 I (eq. 2.24), con W normalizzata per riga.

    att: array [L, n, n] di attenzioni già mediate sulle teste."""
    att = np.asarray(att, dtype=np.float64)
    W = att / np.clip(att.sum(axis=-1, keepdims=True), 1e-12, None)
    n = W.shape[-1]
    return 0.5 * W + 0.5 * np.eye(n)[None, :, :]


def rollout_matrix(att: np.ndarray) -> np.ndarray:
    """Ã(L) = A(L) ··· A(1) (eq. 2.20-2.21)."""
    A = residual_attention(att)
    joint = A[0]
    for layer in A[1:]:
        joint = layer @ joint
    return joint


def rollout_scores(att: np.ndarray, query: int = 0) -> np.ndarray:
    """Rilevanza di ogni token di input per la rappresentazione finale del token
    `query` (0 = [CLS]/<s>, il token usato dalla testa di classificazione)."""
    return rollout_matrix(att)[query]


FLOW_SCALE = 1_000_000  # le capacità sono convertite in interi (1e-6 di risoluzione)


def _flow_graph(A: np.ndarray):
    """Grafo a strati di Abnar e Zuidema: nodo (l, i) -> nodo (l-1, j) con capacità A(l)[i, j]."""
    import scipy.sparse as sp
    L, n, _ = A.shape
    N = (L + 1) * n
    rows, cols, caps = [], [], []
    for l in range(1, L + 1):
        M = A[l - 1]
        i, j = np.nonzero(M > 0)
        rows.append(l * n + i)
        cols.append((l - 1) * n + j)
        caps.append(M[i, j])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    w = np.maximum(1, np.round(np.concatenate(caps) * FLOW_SCALE)).astype(np.int32)
    return sp.csr_matrix((w, (r, c)), shape=(N, N))


def _flow_to_sinks(args):
    from scipy.sparse.csgraph import maximum_flow
    G, source, sinks = args
    return [maximum_flow(G, source, int(t), method="dinic").flow_value / FLOW_SCALE for t in sinks]


def flow_scores(att: np.ndarray, query: int = 0, workers: int = 1) -> np.ndarray:
    """Attention Flow: per ogni token di input i, flusso massimo dal nodo `query`
    dell'ultimo layer al nodo i del layer di input, con capacità date dalle
    attenzioni corrette per il residuo (stessa correzione del Rollout).

    Calcolo esatto (Dinic, scipy). Il costo cresce circa come L·n³: per questo il
    numero di token è limitato a parte (vedi `flow_max_length` in explainers.py)."""
    A = residual_attention(att)
    L, n, _ = A.shape
    G = _flow_graph(A)
    source = L * n + query
    sinks = list(range(n))
    if workers <= 1 or n < 2 * workers:
        return np.array(_flow_to_sinks((G, source, sinks)))
    from concurrent.futures import ProcessPoolExecutor
    chunks = [sinks[k::workers] for k in range(workers)]
    out = np.zeros(n)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chunk, vals in zip(chunks, ex.map(_flow_to_sinks, [(G, source, ch) for ch in chunks])):
            out[chunk] = vals
    return out


# ==== KernelSHAP (Lundberg e Lee 2017; tesi eq. 2.7-2.10) ====
def _kernel_weight(M: int, s: int) -> float:
    """pi_x(z') = (M-1) / (C(M,|z'|) |z'| (M-|z'|))  (eq. 2.10)."""
    from math import comb
    return (M - 1) / (comb(M, s) * s * (M - s))


def sample_coalitions(M: int, nsamples: int, rng: np.random.RandomState):
    """Coalizioni z' in {0,1}^M e relativi pesi per la regressione dell'eq. 2.9.

    Se 2^M - 2 <= nsamples si enumerano tutte le coalizioni (stima esatta dei valori
    di Shapley). Altrimenti campionamento Monte Carlo: la dimensione s si estrae con
    probabilità proporzionale al peso totale delle coalizioni di quella dimensione,
    (M-1)/(s(M-s)), poi un sottoinsieme uniforme di dimensione s; ogni campione ha
    quindi lo stesso peso nella regressione."""
    if M >= 2 and M < 31 and 2 ** M - 2 <= nsamples:
        Z, w = [], []
        for code in range(1, 2 ** M - 1):
            z = np.array([(code >> j) & 1 for j in range(M)], dtype=float)
            Z.append(z)
            w.append(_kernel_weight(M, int(z.sum())))
        return np.array(Z), np.array(w), True
    sizes = np.arange(1, M)
    p = 1.0 / (sizes * (M - sizes))
    p = p / p.sum()
    Z = np.zeros((nsamples, M))
    for k, s in enumerate(rng.choice(sizes, size=nsamples, p=p)):
        Z[k, rng.choice(M, size=int(s), replace=False)] = 1.0
    return Z, np.ones(nsamples), False


def kernel_shap_values(f: Callable[[np.ndarray], np.ndarray], M: int, nsamples: int,
                       rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Valori SHAP phi_1..phi_M di f in x (tutte le feature presenti), con baseline
    f(nessuna feature) = phi_0 (eq. 2.8-2.9).

    f riceve una matrice binaria (una riga per coalizione) e restituisce un valore
    per riga. Si risolve la regressione pesata dell'eq. 2.9 con il vincolo di
    additività sum(phi) = f(x) - phi_0. Con meno campioni che feature il sistema è
    sottodeterminato e si prende la soluzione di norma minima."""
    if M == 0:
        return np.zeros(0)
    rng = rng if rng is not None else np.random.mtrand._rand
    ends = np.asarray(f(np.vstack([np.zeros(M), np.ones(M)])), dtype=float).reshape(-1)
    phi0, fx = ends[0], ends[1]
    if M == 1:
        return np.array([fx - phi0])
    Z, w, _ = sample_coalitions(M, nsamples, rng)
    y = np.asarray(f(Z), dtype=float).reshape(-1) - phi0
    # KKT: min (y - Z phi)' W (y - Z phi)  s.t.  1' phi = fx - phi0
    XtW = Z.T * w
    K = np.zeros((M + 1, M + 1))
    K[:M, :M] = 2 * XtW @ Z
    K[:M, M] = 1.0
    K[M, :M] = 1.0
    rhs = np.concatenate([2 * XtW @ y, [fx - phi0]])
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return sol[:M]


# ==== LIME (Ribeiro et al. 2016; tesi eq. 2.6) ====
def lime_word_scores(words: List[str], classifier_fn: Callable, mask_string: str,
                     num_samples: int) -> np.ndarray:
    """Pesi LIME per la classe positiva, uno per parola (in posizione).

    bow=False: ogni occorrenza è una feature distinta e le parole rimosse sono
    sostituite da `mask_string`. Il campionamento usa np.random (random_state=None),
    quindi è controllato dai seed fissati dal chiamante."""
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=["negative", "positive"], bow=False,
                                  split_expression=lambda s: s.split(" "),
                                  mask_string=mask_string, random_state=None)
    exp = explainer.explain_instance(" ".join(words), classifier_fn, labels=(1,),
                                     num_features=len(words), num_samples=num_samples)
    n_feat = exp.domain_mapper.indexed_string.num_words()
    if n_feat != len(words):
        raise RuntimeError(f"LIME ha {n_feat} feature per {len(words)} parole")
    scores = np.zeros(len(words))
    for feat, weight in exp.as_map()[1]:
        scores[feat] = weight
    return scores
