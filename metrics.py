"""
metrics_simplified.py – Metriche XAI Evalutation
================================================================
Metriche implementate:
- Robustness: stabilità sotto perturbazioni
- Consistency: stabilità con inference seed diversi
- Contrastivity: diversità tra classi opposte
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Callable, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from scipy.stats import spearmanr, entropy
from tqdm import tqdm
from explainers import Attribution
import models

# ==== Parametri configurazione ====
DEFAULT_PERTURBATION_RATIO = 0.15
MIN_SHARED_TOKENS = 2
RANDOM_STATE = 42

# Parametri consistency
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]
MAX_CONSISTENCY_SAMPLES = 100  # Bilanciamento tra accuratezza e velocità

# ==== Memory Management ====
def clear_memory_if_needed():
    """Cleanup memoria se necessario."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        if allocated > 8.0:
            models.clear_gpu_memory()

# ==== Perturbation Functions ====
def _random_mask(text: str, ratio: float = DEFAULT_PERTURBATION_RATIO, mask_token: str = "[MASK]") -> str:
    """Maschera parole casuali."""
    if not text.strip():
        return text
    
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    
    n_to_mask = max(1, int(len(tokens) * ratio))
    n_to_mask = min(n_to_mask, len(tokens) - 1)
    
    try:
        idx_to_mask = random.sample(range(len(tokens)), n_to_mask)
        for i in idx_to_mask:
            tokens[i] = mask_token
        return " ".join(tokens)
    except Exception:
        return text

def _random_delete(text: str, ratio: float = DEFAULT_PERTURBATION_RATIO) -> str:
    """Elimina parole casuali."""
    if not text.strip():
        return text
    
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    
    n_to_delete = max(1, int(len(tokens) * ratio))
    n_to_delete = min(n_to_delete, len(tokens) - 1)
    
    try:
        idx_to_keep = random.sample(range(len(tokens)), len(tokens) - n_to_delete)
        idx_to_keep.sort()
        return " ".join(tokens[i] for i in idx_to_keep)
    except Exception:
        return text

def _random_substitute(text: str, ratio: float = DEFAULT_PERTURBATION_RATIO) -> str:
    """Sostituisce parole con sinonimi."""
    substitutions = {
        "good": "great", "bad": "terrible", "nice": "pleasant", "awful": "horrible",
        "love": "like", "hate": "dislike", "amazing": "wonderful", "terrible": "bad",
        "excellent": "good", "poor": "bad", "fantastic": "great", "boring": "dull"
    }
    
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    
    n_to_substitute = max(1, int(len(tokens) * ratio))
    
    try:
        idx_to_substitute = random.sample(range(len(tokens)), min(n_to_substitute, len(tokens)))
        for i in idx_to_substitute:
            word = tokens[i].lower()
            if word in substitutions:
                tokens[i] = substitutions[word]
        return " ".join(tokens)
    except Exception:
        return text

PERTURBATION_FUNCTIONS = [_random_mask, _random_delete, _random_substitute]

# ==== 1. ROBUSTNESS ====
def compute_robustness(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    text: str,
    n_perturbations: int = 2
) -> float:
    """Calcola robustness con perturbazioni."""
    try:
        # Attribution originale
        orig_attr = explainer(text)
        if not orig_attr.tokens or not orig_attr.scores:
            return 0.0
        
        all_diffs = []
        
        for perturb_func in PERTURBATION_FUNCTIONS:
            for _ in range(n_perturbations):
                try:
                    # Perturbazione
                    pert_text = perturb_func(text)
                    if pert_text == text:
                        continue
                    
                    # Attribution perturbata
                    pert_attr = explainer(pert_text)
                    if not pert_attr.tokens or not pert_attr.scores:
                        continue
                    
                    # Calcola differenze per token condivisi
                    score_diffs = []
                    for tok, score in zip(orig_attr.tokens, orig_attr.scores):
                        if tok in pert_attr.tokens:
                            j = pert_attr.tokens.index(tok)
                            diff = abs(score - pert_attr.scores[j])
                            score_diffs.append(diff)
                    
                    if score_diffs:
                        all_diffs.extend(score_diffs)
                        
                except Exception:
                    continue
        
        return float(np.mean(all_diffs)) if all_diffs else 0.0
        
    except Exception:
        return 0.0

def evaluate_robustness_over_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    show_progress: bool = True,
) -> float:
    """Valuta robustness su dataset."""
    try:
        robustness_scores = []
        iterator = tqdm(texts, desc="Robustness", leave=False) if show_progress else texts
        
        for i, text in enumerate(iterator):
            try:
                if text.strip():
                    score = compute_robustness(model, tokenizer, explainer, text)
                    robustness_scores.append(score)
                
                # Cleanup periodico
                if i % 50 == 0:
                    clear_memory_if_needed()
                    
            except Exception:
                continue
        
        return float(np.mean(robustness_scores)) if robustness_scores else 0.0
        
    except Exception:
        return 0.0

# ==== 2. CONSISTENCY (CORRECT LOGIC) ====
def _compute_single_observation_correlation(attr_a: Attribution, attr_b: Attribution) -> float:
    """Calcola correlazione di Spearman tra due explanations della STESSA osservazione."""
    
    # Check explanations vuote
    if not attr_a.tokens or not attr_b.tokens:
        return 0.1  # Fallback: correlazione bassa ma non zero
    
    # Token matching flessibile (case-insensitive)
    shared_scores_a, shared_scores_b = [], []
    
    tokens_b_dict = {tok.lower(): i for i, tok in enumerate(attr_b.tokens)}
    
    for token, score_a in zip(attr_a.tokens, attr_a.scores):
        token_key = token.lower()
        if token_key in tokens_b_dict:
            idx = tokens_b_dict[token_key]
            shared_scores_a.append(score_a)
            shared_scores_b.append(attr_b.scores[idx])
    
    # Soglia: almeno 1 token condiviso
    if len(shared_scores_a) >= 1:
        arr_a = np.array(shared_scores_a)
        arr_b = np.array(shared_scores_b)
        
        # Check validità
        if np.any(np.isnan(arr_a)) or np.any(np.isnan(arr_b)):
            return 0.0
        if np.any(np.isinf(arr_a)) or np.any(np.isinf(arr_b)):
            return 0.0
        
        # Check varianza
        if np.var(arr_a) < 1e-12 and np.var(arr_b) < 1e-12:
            return 1.0 if np.allclose(arr_a, arr_b) else 0.0
        elif np.var(arr_a) < 1e-12 or np.var(arr_b) < 1e-12:
            return 0.0
        
        # Calcolo Spearman
        try:
            rho, _ = spearmanr(arr_a, arr_b)
            return float(rho) if not np.isnan(rho) else 0.0
        except Exception:
            return 0.0
    else:
        return 0.05  # Correlazione molto bassa

def compute_consistency_inference_seed_CORRECT(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = True
) -> Tuple[float, float]:
    """
    LOGICA CORRETTA per Consistency:
    
    1. Per ogni osservazione i:
       - Genera explanations con 4 seed diversi
       - Calcola 6 correlazioni tra coppie di seed (C(4,2) = 6)
       - Calcola MEDIA delle 6 correlazioni → spearman_mean_i
    
    2. Su tutto il dataset (N osservazioni):
       - Array: [spearman_mean_1, spearman_mean_2, ..., spearman_mean_N]
       - MEAN = Σ(spearman_mean_i) / N
       - STD = std([spearman_mean_1, spearman_mean_2, ..., spearman_mean_N])
    
    3. Tabella finale: MEAN ± STD
    
    Limitato a MAX_CONSISTENCY_SAMPLES = 100 
    """
    
    # Applica limite per gestire complessità computazionale O(n × seeds²)
    if len(texts) > MAX_CONSISTENCY_SAMPLES:
        print(f"[CONSISTENCY] Limiting to {MAX_CONSISTENCY_SAMPLES} samples (from {len(texts)}) for computational efficiency")
        texts = texts[:MAX_CONSISTENCY_SAMPLES]
    
    if show_progress:
        print(f"[CONSISTENCY-CORRECT] {len(seeds)} seeds, {len(texts)} observations")
        print(f"[CONSISTENCY-CORRECT] Will compute {len(seeds)*(len(seeds)-1)//2} correlations per observation")
        estimated_time = len(texts) * 0.8  # ~0.8 min per observation
        print(f"[CONSISTENCY-CORRECT] Estimated time: ~{estimated_time:.1f} minutes")
    
    # Setup modello per dropout
    original_mode = model.training
    model.train()
    
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
    
    try:
        # Array per memorizzare spearman_mean per ogni osservazione
        per_observation_mean_spearman = []
        
        # LOOP PRINCIPALE: Per ogni osservazione
        for obs_idx, text in enumerate(texts):
            if show_progress and obs_idx % 20 == 0:  # Progress ogni 20
                print(f"  [OBS] {obs_idx + 1}/{len(texts)}")
            
            # STEP 1: Genera explanations per tutti i seed per questa osservazione
            obs_explanations = {}
            
            for seed in seeds:
                # Set seed per dropout
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                try:
                    attr = explainer(text)
                    obs_explanations[seed] = attr
                except Exception:
                    obs_explanations[seed] = Attribution([], [])
            
            # STEP 2: Calcola correlazioni tra tutte le coppie di seed per questa osservazione
            obs_correlations = []
            
            for i, seed_a in enumerate(seeds):
                for seed_b in seeds[i+1:]:
                    attr_a = obs_explanations[seed_a]
                    attr_b = obs_explanations[seed_b]
                    
                    # Calcola correlazione
                    correlation = _compute_single_observation_correlation(attr_a, attr_b)
                    
                    # Aggiungi solo se valida
                    if not np.isnan(correlation) and not np.isinf(correlation):
                        obs_correlations.append(correlation)
            
            # STEP 3: Calcola media delle correlazioni per questa osservazione
            if obs_correlations:
                spearman_mean_i = np.mean(obs_correlations)
            else:
                spearman_mean_i = 0.1  # Valore neutro basso
                
            per_observation_mean_spearman.append(spearman_mean_i)
            
            # Cleanup periodico ogni 25 osservazioni
            if obs_idx > 0 and obs_idx % 25 == 0:
                clear_memory_if_needed()
        
        # STEP 4: Calcola statistiche finali su tutto il dataset
        if per_observation_mean_spearman:
            # QUESTA È LA LOGICA CORRETTA
            final_mean = np.mean(per_observation_mean_spearman)  # Media degli spearman_mean_i
            final_std = np.std(per_observation_mean_spearman, ddof=1)  # Std degli spearman_mean_i
        else:
            final_mean = 0.0
            final_std = 0.0
        
        if show_progress:
            print(f"  [RESULT] Final: {final_mean:.4f} ± {final_std:.4f}")
            print(f"  [RESULT] Based on {len(per_observation_mean_spearman)} observations")
        
        return final_mean, final_std
        
    finally:
        # Ripristina stato modello
        model.train(original_mode)
        for name, param in model.named_parameters():
            param.requires_grad_(original_requires_grad[name])

def evaluate_consistency_over_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = True
) -> Tuple[float, float]:
    """Wrapper per consistency evaluation che restituisce (media, std)."""
    return compute_consistency_inference_seed_CORRECT(
        model=model,
        tokenizer=tokenizer,
        explainer=explainer,
        texts=texts,
        seeds=seeds,
        show_progress=show_progress
    )

# ==== 3. CONTRASTIVITY ====
def _normalize_scores_for_distribution(scores: List[float]) -> np.ndarray:
    """Normalizza scores per distribuzione di probabilità."""
    arr = np.array(scores, dtype=float)
    
    # Sposta a valori non-negativi
    arr = arr - np.min(arr)
    
    # Normalizza
    total = np.sum(arr)
    if total == 0:
        return np.ones(len(arr)) / len(arr)
    
    return arr / total

def compute_contrastivity(
    positive_attrs: List[Attribution],
    negative_attrs: List[Attribution],
    use_jensen_shannon: bool = False,
) -> float:
    """Calcola contrastivity con gestione memoria."""
    try:
        if not positive_attrs or not negative_attrs:
            return 0.0
        
        # Accumula token scores per classe
        token_scores_pos = {}
        token_scores_neg = {}
        
        def accumulate_scores(score_dict: dict, attr: Attribution):
            for tok, score in zip(attr.tokens, attr.scores):
                if tok.strip() and tok not in ['[CLS]', '[SEP]', '[PAD]', '[ERROR]']:
                    score_dict[tok] = score_dict.get(tok, 0.0) + score
        
        # Accumula scores
        for attr in positive_attrs:
            accumulate_scores(token_scores_pos, attr)
        for attr in negative_attrs:
            accumulate_scores(token_scores_neg, attr)
        
        # Vocabolario unificato
        vocab = set(token_scores_pos.keys()) | set(token_scores_neg.keys())
        if len(vocab) < 2:
            return 0.0
        
        vocab = sorted(vocab)[:1000]  # Limita per memoria
        
        # Distribuzioni
        pos_scores = [token_scores_pos.get(tok, 0.0) for tok in vocab]
        neg_scores = [token_scores_neg.get(tok, 0.0) for tok in vocab]
        
        p = _normalize_scores_for_distribution(pos_scores)
        q = _normalize_scores_for_distribution(neg_scores)
        
        # Calcola divergenza
        if use_jensen_shannon:
            m = 0.5 * (p + q)
            js_div = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)
            return float(js_div)
        else:
            epsilon = 1e-10
            q_smooth = q + epsilon
            kl_div = entropy(p, q_smooth, base=2)
            return float(kl_div)
            
    except Exception:
        return 0.0

# ==== Batch Processing Utilities ====
def process_attributions_batch(
    texts: List[str], 
    explainer: Callable[[str], Attribution],
    batch_size: int = 10,
    show_progress: bool = True
) -> List[Attribution]:
    """Processa attributions in batch con memory management."""
    results = []
    
    # Dividi in batch
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    iterator = tqdm(batches, desc="Processing batch", leave=False) if show_progress else batches
    
    for batch in iterator:
        batch_results = []
        for text in batch:
            try:
                attr = explainer(text)
                batch_results.append(attr)
            except Exception:
                batch_results.append(Attribution([], []))
        
        results.extend(batch_results)
        
        # Cleanup ogni batch
        clear_memory_if_needed()
    
    return results

# ==== Test Function (minimal) ====
if __name__ == "__main__":
    print("Testing simplified metrics...")
    
    try:
        import models
        model = models.load_model("tinybert")
        tokenizer = models.load_tokenizer("tinybert")
        
        # Mock explainer per test
        def mock_explainer(text):
            tokens = text.split()[:5]
            scores = np.random.rand(len(tokens)).tolist()
            return Attribution(tokens, scores)
        
        test_texts = [
            "This movie is great!",
            "This movie is terrible.",
            "An okay film."
        ]
        
        print("\nTesting Robustness...")
        robustness = evaluate_robustness_over_dataset(
            model, tokenizer, mock_explainer, test_texts, show_progress=False
        )
        print(f"Robustness: {robustness:.4f}")
        
        print("\nTesting Consistency (CORRECT LOGIC)...")
        mean_consistency, std_consistency = evaluate_consistency_over_dataset(
            model, tokenizer, mock_explainer, test_texts[:2], 
            seeds=[42, 123], show_progress=False
        )
        print(f"Consistency: {mean_consistency:.4f} ± {std_consistency:.4f}")
        
        print("\nTesting Contrastivity...")
        pos_attrs = [mock_explainer(test_texts[0])]
        neg_attrs = [mock_explainer(test_texts[1])]
        contrastivity = compute_contrastivity(pos_attrs, neg_attrs)
        print(f"Contrastivity: {contrastivity:.4f}")
        
        print("\n Simplified metrics test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        models.clear_gpu_memory()