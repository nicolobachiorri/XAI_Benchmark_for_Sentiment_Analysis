"""
explainers.py – I sei metodi XAI come descritti nella tesi (par. 2.5.3)
======================================================================

Ogni explainer è una funzione `testo -> Attribution` (un punteggio per parola, vedi
attribution.py). Regole comuni a tutti:

- il testo va passato con spazi singoli (attribution.normalize_whitespace);
- la classe spiegata è quella positiva (indice da models.positive_index);
- gli explainer **non** cambiano la modalità del modello: è il chiamante a usare
  model.eval() (robustness, contrastivity) o model.train() (consistency, dropout
  attivo). Nella versione della tesi quattro explainer chiamavano model.eval() e
  spegnevano il dropout richiesto dalla consistency;
- il testo è troncato a `max_length` token (512 = limite dei modelli); Attention
  Flow ha un limite separato, `flow_max_length`, per il costo del flusso massimo.

Metodi
    grad_input          Input×Gradient (eq. 2.11-2.12) sugli embedding delle parole
    lrp                 LRP con AH-rule e LN-rule di Ali et al. (eq. 2.18-2.19)
    attention_rollout   Rollout con correzione del residuo (eq. 2.20-2.24), riga del [CLS]
    attention_flow      flusso massimo dal [CLS] dell'ultimo layer a ogni token (pp. 39-40)
    lime                LIME testuale, parole come feature (pp. 31-33)
    shap                KernelSHAP con 15 coalizioni campionate (pp. 33-34), implementato
                        direttamente dalle eq. 2.9-2.10 (attribution.kernel_shap_values)
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import models
from attribution import (Attribution, aggregate_to_words, flow_scores, kernel_shap_values,
                         lime_word_scores, rollout_scores, token_word_ids, word_spans)


# ==== Configurazione ====
@dataclass
class ExplainerConfig:
    max_length: int = 512          # token, limite dei modelli
    flow_max_length: int = 128     # token per Attention Flow (costo ~ L·n³)
    flow_workers: int = 1          # processi per il flusso massimo
    lime_samples: int = 500        # la tesi non lo specifica (default della libreria: 5000)
    shap_samples: int = 15         # tesi, p. 34
    batch_size: int = 32           # testi per forward nelle funzioni di LIME e SHAP
    amp: bool = True               # float16 (autocast) nei forward senza gradiente, solo su GPU


# ==== Tokenizzazione e allineamento token-parole ====
def _encode(text: str, tokenizer, max_length: int):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length,
                    return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    words = text.split(" ")
    wid = token_word_ids(offsets, word_spans(text))
    return models.move_batch_to_device(dict(enc)), words, wid


def visible_text(text: str, tokenizer, max_length: int) -> str:
    """Il testo ridotto alle parole che entrano in `max_length` token."""
    _, words, wid = _encode(text, tokenizer, max_length)
    n = max([w for w in wid if w >= 0], default=-1) + 1
    return " ".join(words[:n])


def _attribution(words: List[str], word_ids: List[int], token_scores) -> Attribution:
    scores = aggregate_to_words(np.asarray(token_scores, dtype=float), word_ids)
    return Attribution(words[:len(scores)], scores.tolist(), n_words_total=len(words))


def _model_inputs(model, enc: dict, with_ids: bool = True) -> dict:
    """Input del forward: DistilBERT non accetta token_type_ids."""
    keys = ["attention_mask"] + (["input_ids"] if with_ids else [])
    if model.config.model_type != "distilbert":
        keys.append("token_type_ids")
    return {k: enc[k] for k in keys if k in enc}


# ==== Input×Gradient ====
def _grad_times_input(model, tokenizer, text: str, cfg: ExplainerConfig) -> Attribution:
    enc, words, wid = _encode(text, tokenizer, cfg.max_length)
    pos = models.positive_index(model)
    emb = model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
    out = model(inputs_embeds=emb, **_model_inputs(model, enc, with_ids=False))
    target = out.logits[0, pos]
    (grad,) = torch.autograd.grad(target, emb)
    tok = (emb * grad).sum(-1)[0].detach().float().cpu().numpy()
    return _attribution(words, wid, tok)


def _grad_input(model, tokenizer, cfg: ExplainerConfig):
    def explain(text: str) -> Attribution:
        return _grad_times_input(model, tokenizer, text, cfg)
    return explain


# ==== LRP (Ali et al., AH-rule e LN-rule) ====
_ORIG_SOFTMAX = F.softmax
_ORIG_LAYER_NORM = F.layer_norm


def _softmax_const(*args, **kwargs):
    # AH-rule: i pesi di attenzione p_ij sono costanti rispetto all'input
    return _ORIG_SOFTMAX(*args, **kwargs).detach()


def _layer_norm_const(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # LN-rule: il fattore di scala (eps + Var[x])^-1/2 è costante, la centratura no
    dims = tuple(range(-len(normalized_shape), 0))
    mean = input.mean(dim=dims, keepdim=True)
    var = input.var(dim=dims, unbiased=False, keepdim=True)
    y = (input - mean) / torch.sqrt(var + eps).detach()
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


@contextmanager
def lrp_rules():
    """Durante il blocco, softmax e LayerNorm propagano la rilevanza secondo le regole
    AH e LN: Gradient×Input sul grafo così modificato è LRP-0 con le due regole
    (Ali et al. 2022, sez. 3). Il resto della rete (strati lineari, GELU, tanh della
    testa) è trattato dal gradiente, come nel loro lavoro."""
    torch.nn.functional.softmax = _softmax_const
    torch.nn.functional.layer_norm = _layer_norm_const
    try:
        yield
    finally:
        torch.nn.functional.softmax = _ORIG_SOFTMAX
        torch.nn.functional.layer_norm = _ORIG_LAYER_NORM


def _check_eager(model):
    impl = getattr(model.config, "_attn_implementation", "eager")
    if impl not in (None, "eager"):
        raise RuntimeError(f"LRP, Rollout e Flow richiedono attn_implementation='eager' (trovato {impl!r})")


def _lrp(model, tokenizer, cfg: ExplainerConfig):
    _check_eager(model)

    def explain(text: str) -> Attribution:
        with lrp_rules():
            return _grad_times_input(model, tokenizer, text, cfg)
    return explain


# ==== Attention Rollout e Attention Flow ====
def _attentions(model, enc) -> np.ndarray:
    """Attenzioni [L, n, n] mediate sulle teste (in modalità train includono il dropout)."""
    with torch.no_grad():
        out = model(**_model_inputs(model, enc), output_attentions=True)
    if not out.attentions:
        raise RuntimeError("il modello non restituisce le attenzioni (serve attn_implementation='eager')")
    return torch.stack([a[0].float().mean(0) for a in out.attentions]).cpu().numpy()


def _attention_rollout(model, tokenizer, cfg: ExplainerConfig):
    _check_eager(model)

    def explain(text: str) -> Attribution:
        enc, words, wid = _encode(text, tokenizer, cfg.max_length)
        return _attribution(words, wid, rollout_scores(_attentions(model, enc), query=0))
    return explain


def _attention_flow(model, tokenizer, cfg: ExplainerConfig):
    _check_eager(model)

    def explain(text: str) -> Attribution:
        enc, words, wid = _encode(text, tokenizer, cfg.flow_max_length)
        return _attribution(words, wid, flow_scores(_attentions(model, enc), query=0, workers=cfg.flow_workers))
    return explain


# ==== Funzione di probabilità per LIME e SHAP ====
def _proba_fn(model, tokenizer, cfg: ExplainerConfig) -> Callable[[List[str]], np.ndarray]:
    pos = models.positive_index(model)
    use_amp = cfg.amp and models.DEVICE.type == "cuda"

    def proba(texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), cfg.batch_size):
            enc = tokenizer(list(texts[i:i + cfg.batch_size]), return_tensors="pt", padding=True,
                            truncation=True, max_length=cfg.max_length)
            enc = models.move_batch_to_device(dict(enc))
            amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
            with torch.no_grad(), amp_ctx:
                logits = model(**_model_inputs(model, enc)).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, pos].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
    return proba


def _truncate_words(text: str, tokenizer, cfg: ExplainerConfig) -> List[str]:
    return visible_text(text, tokenizer, cfg.max_length).split(" ")


# ==== LIME ====
def _lime_text(model, tokenizer, cfg: ExplainerConfig):
    proba = _proba_fn(model, tokenizer, cfg)
    mask = tokenizer.mask_token or tokenizer.unk_token

    def classifier_fn(texts):
        p = proba(list(texts))
        return np.stack([1.0 - p, p], axis=1)

    def explain(text: str) -> Attribution:
        words = _truncate_words(text, tokenizer, cfg)
        scores = lime_word_scores(words, classifier_fn, mask, cfg.lime_samples)
        return Attribution(words, scores.tolist(), n_words_total=len(text.split(" ")))
    return explain


# ==== KernelSHAP ====
def _kernel_shap(model, tokenizer, cfg: ExplainerConfig):
    proba = _proba_fn(model, tokenizer, cfg)
    mask = tokenizer.mask_token or tokenizer.unk_token

    def explain(text: str) -> Attribution:
        words = _truncate_words(text, tokenizer, cfg)

        def f(Z):
            texts = [" ".join(w if z > 0.5 else mask for w, z in zip(words, row)) for row in np.atleast_2d(Z)]
            return proba(texts)

        # baseline: tutte le parole sostituite dal token di mask (p. 34);
        # `shap_samples` coalizioni campionate (15 nella tesi)
        vals = kernel_shap_values(f, len(words), cfg.shap_samples)
        return Attribution(words, vals.tolist(), n_words_total=len(text.split(" ")))
    return explain


# ==== Factory ====
_EXPLAINERS = {
    "lime": _lime_text,
    "shap": _kernel_shap,
    "grad_input": _grad_input,
    "attention_rollout": _attention_rollout,
    "attention_flow": _attention_flow,
    "lrp": _lrp,
}


def list_explainers() -> List[str]:
    return list(_EXPLAINERS)


def get_explainer(name: str, model, tokenizer, cfg: Optional[ExplainerConfig] = None) -> Callable[[str], Attribution]:
    if name not in _EXPLAINERS:
        raise ValueError(f"Explainer '{name}' non disponibile: {list(_EXPLAINERS)}")
    return _EXPLAINERS[name](model, tokenizer, cfg or ExplainerConfig())
