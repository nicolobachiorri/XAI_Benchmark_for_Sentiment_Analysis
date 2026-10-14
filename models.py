"""
models.py – I cinque modelli fine-tuned della tesi (cap. 3.2, tabella 3.1)
=========================================================================

Cambiamenti rispetto alla versione della tesi:
- i pesi sono caricati in float32 (prima float16 su GPU): i gradienti di
  Input×Gradient e LRP in float16 rischiano underflow. Il float16 resta disponibile
  solo nei forward senza gradiente di LIME e SHAP (explainers.ExplainerConfig.amp);
- il numero di classi viene dal config del modello: prima si forzava num_labels=2
  con ignore_mismatched_sizes=True, che con una testa di dimensione diversa
  l'avrebbe reinizializzata a caso senza errore;
- attn_implementation="eager" è obbligatoria: serve per leggere le attenzioni
  (Rollout, Flow) e perché l'LRP intercetta la softmax dell'attenzione;
- i parametri non richiedono gradiente (si deriva solo rispetto all'input).
"""

import gc
import os
from typing import Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==== Modelli (Hugging Face) ====
MODELS: Dict[str, str] = {
    "tinybert":      "Harsha901/tinybert-imdb-sentiment-analysis-model",
    "distilbert":    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "roberta-base":  "AnkitAI/reviews-roberta-base-sentiment-analysis",
    "bert-large":    "assemblyai/bert-large-uncased-sst2",
    "roberta-large": "siebert/sentiment-roberta-large-english",
}


def register_model(key: str, path: str) -> None:
    """Aggiunge un modello locale o di Hugging Face (usato dai test con modelli finti)."""
    MODELS[key] = path


# ==== Device ====
def _pick_device() -> torch.device:
    forced = os.environ.get("XAI_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _pick_device()


def set_device(name: str) -> None:
    global DEVICE
    DEVICE = torch.device(name)


def clear_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==== Caricamento ====
def load_tokenizer(model_key: str):
    tok = AutoTokenizer.from_pretrained(MODELS[model_key], use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(f"{model_key}: serve un tokenizer 'fast' (offset dei token)")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def load_model(model_key: str):
    clear_gpu_memory()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[model_key], attn_implementation="eager")
    model = model.float()  # float32 qualunque sia il dtype del checkpoint
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(DEVICE)
    model.eval()
    positive_index(model)  # verifica subito che la classe positiva sia riconoscibile
    return model


def positive_index(model) -> int:
    """Indice della classe positiva, dal config del modello (id2label)."""
    id2label = {int(k): str(v).lower() for k, v in getattr(model.config, "id2label", {}).items()}
    for i, lab in id2label.items():
        if lab.startswith("pos"):
            return i
    if model.config.num_labels == 2:
        return 1  # convenzione SST-2/IMDB: LABEL_1 = positivo
    raise ValueError(f"classe positiva non riconoscibile da id2label={id2label}")


def move_batch_to_device(batch):
    if isinstance(batch, dict):
        return {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in batch.items()}
    return batch.to(DEVICE) if hasattr(batch, "to") else batch
