"""
models.py – Gestione modelli pre-trained per sentiment analysis 
============================================================

Assume sempre ambiente Colab con GPU disponibile.
"""

import os
import gc
import warnings
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==== Mappa modelli pre-trained ====
MODELS: Dict[str, str] = {
    "distilbert":       "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "roberta-large":    "siebert/sentiment-roberta-large-english", 
    "bert-large":       "assemblyai/bert-large-uncased-sst2",
    "roberta-base":     "AnkitAI/reviews-roberta-base-sentiment-analysis",
    "tinybert":         "Harsha901/tinybert-imdb-sentiment-analysis-model",
}

# ==== GPU Configuration ====
def setup_colab_gpu():
    """Setup GPU per Colab."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] Using: {torch.cuda.get_device_name()}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Ottimizzazioni Colab
        torch.cuda.set_per_process_memory_fraction(0.9)  # Usa 90% GPU memory
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        return device
    else:
        print("[WARNING] GPU not available, using CPU")
        return torch.device("cpu")

# Device globale
DEVICE = setup_colab_gpu()

# ==== Memory Management ====
def get_gpu_memory_usage():
    """Restituisce uso memoria GPU in GB."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3
        }
    return {"allocated": 0, "reserved": 0, "max_allocated": 0}

def clear_gpu_memory():
    """Pulisce memoria GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ==== Tokenizer Loading ====
def load_tokenizer(model_key: str):
    """Carica tokenizer."""
    if model_key not in MODELS:
        raise ValueError(f"Modello '{model_key}' non trovato. Disponibili: {list(MODELS.keys())}")
    
    model_name = MODELS[model_key]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix pad_token se mancante
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
            
        return tokenizer
    except Exception as e:
        print(f"Errore caricamento tokenizer {model_key}: {e}")
        raise

# ==== Model Loading ====
def load_model(model_key: str, num_labels: int = 2):
    """Carica modello su GPU con gestione OOM."""
    if model_key not in MODELS:
        raise ValueError(f"Modello '{model_key}' non trovato. Disponibili: {list(MODELS.keys())}")
    
    model_name = MODELS[model_key]
    
    try:
        # Cleanup preventivo
        clear_gpu_memory()
        
        # Carica modello
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                attn_implementation="eager",  # Più stabile per Colab
                low_cpu_mem_usage=True,
            )
        except (ValueError, TypeError):
            # Fallback senza attn_implementation
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
        
        # Sposta su GPU
        model = model.to(DEVICE)
        model.eval()
        
        # Log memoria
        if DEVICE.type == "cuda":
            gpu_mem = get_gpu_memory_usage()
            print(f"[GPU] Model loaded: {gpu_mem['allocated']:.2f}GB allocated")
        
        return model
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] GPU out of memory per {model_key}. Prova un modello più piccolo.")
            clear_gpu_memory()
            raise RuntimeError(f"OOM loading {model_key}. Try smaller model like 'tinybert' or 'distilbert'")
        raise
    except Exception as e:
        print(f"Errore caricamento modello {model_key}: {e}")
        raise

# ==== Batch Processing ====
def move_batch_to_device(batch):
    """Sposta batch su GPU."""
    if isinstance(batch, dict):
        return {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in batch.items()}
    elif hasattr(batch, 'to'):
        return batch.to(DEVICE)
    else:
        return batch

# ==== Utilities ====
def print_gpu_status():
    """Stampa status GPU."""
    if torch.cuda.is_available():
        mem = get_gpu_memory_usage()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {mem['allocated']:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("GPU: Not available")

# ==== Test ====
if __name__ == "__main__":
    """Test veloce per Colab."""
    print("Testing models on Colab...")
    print_gpu_status()
    
    # Test modelli piccoli
    for model_key in ["tinybert", "distilbert"]:
        try:
            print(f"\nTesting {model_key}...")
            
            # Tokenizer
            tokenizer = load_tokenizer(model_key)
            print(f"  Tokenizer: ")
            
            # Model
            model = load_model(model_key)
            print(f"  Model: ")
            
            # Inference test
            test_text = "This is a test."
            encoded = tokenizer(test_text, return_tensors="pt", max_length=50, truncation=True)
            encoded = move_batch_to_device(encoded)
            
            with torch.no_grad():
                outputs = model(**encoded)
            
            print(f"  Inference:  (shape: {outputs.logits.shape})")
            print(f"  SUCCESS: {model_key}")
            
            # Cleanup
            del model
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  FAILED: {model_key} - {e}")
    
    print("\nTest completed!")
    print_gpu_status()

    