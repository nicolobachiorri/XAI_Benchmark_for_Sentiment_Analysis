"""
metrics.py – Metriche XAI ottimizzate per Google Colab con dataset clusterizzato
============================================================

OTTIMIZZAZIONI PER COLAB:
1. Adattato per dataset ridotto (400 esempi)
2. Memory-efficient computation
3. Progress tracking ottimizzato
4. Inference seed consistency più veloce
5. Batch processing per GPU efficiency

Metriche implementate:
- Robustness: stabilità sotto perturbazioni
- Consistency: stabilità con inference seed diversi  
- Contrastivity: diversità tra classi opposte
- Human Reasoning: accordo con ranking human-like (integrato)
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Callable, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from scipy.stats import spearmanr, entropy
from tqdm import tqdm
from explainers import Attribution
import models

# ==== Parametri ottimizzati per Colab ====
DEFAULT_PERTURBATION_RATIO = 0.15
MIN_SHARED_TOKENS = 2
RANDOM_STATE = 42

# Parametri consistency (ottimizzati per Colab)
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]  # 4 seed come richiesto
MAX_CONSISTENCY_SAMPLES = 50  # Limite per consistency

# ==== Memory-efficient Helper Functions ====
def clear_memory_if_needed():
    """Cleanup memoria se necessario."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        if allocated > 8.0:  # Se >8GB, cleanup
            models.clear_gpu_memory()

# ==== Perturbation Functions (ottimizzate) ====
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

# ==== 1. ROBUSTNESS (ottimizzata) ====
def compute_robustness(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    text: str,
    n_perturbations: int = 2  # Ridotto da 3 per velocità
) -> float:
    """Calcola robustness con meno perturbazioni."""
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
    """Valuta robustness su dataset con progress tracking."""
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

# ==== 2. CONSISTENCY (ottimizzata con inference seed) ====
"""
MODIFICHE PER CONSISTENCY CON DEVIAZIONE STANDARD
================================================

Modifica la funzione compute_consistency_inference_seed in metrics.py
per restituire sia media che deviazione standard delle correlazioni.
"""

# ==== 2. CONSISTENCY (MODIFICATA per restituire media ± std) ====
def compute_consistency_inference_seed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = True
) -> Tuple[float, float]:  # CAMBIATO: restituisce (media, std)
    """Consistency con inference seed (MODIFICATA per restituire media ± std)."""
    
    # Limita testi per consistency (computazionalmente intensiva)
    if len(texts) > MAX_CONSISTENCY_SAMPLES:
        texts = texts[:MAX_CONSISTENCY_SAMPLES]
        if show_progress:
            print(f"[CONSISTENCY] Limited to {MAX_CONSISTENCY_SAMPLES} samples for efficiency")
    
    if show_progress:
        print(f"[CONSISTENCY] Computing with {len(seeds)} seeds on {len(texts)} texts...")
    
    # Attiva training mode per dropout
    original_mode = model.training
    model.train()
    
    # Disabilita gradients
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
    
    try:
        # Genera explanations per ogni seed
        explanations_by_seed = {}
        
        for seed in seeds:
            if show_progress:
                print(f"  [SEED] Processing {seed}...")
            
            explanations = []
            text_iterator = tqdm(texts, desc=f"Seed {seed}", leave=False) if show_progress else texts
            
            for text in text_iterator:
                try:
                    # Set seed per dropout stocastico
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    
                    attr = explainer(text)
                    explanations.append(attr)
                    
                except Exception:
                    explanations.append(Attribution([], []))
            
            explanations_by_seed[seed] = explanations
            
            # Cleanup dopo ogni seed
            clear_memory_if_needed()
        
        # Calcola correlazioni tra tutte le coppie di seed
        correlations = []
        
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i+1:]:
                if show_progress:
                    print(f"  [CORR] Computing {seed_a} vs {seed_b}...")
                
                correlation = _compute_spearman_correlation_explanations(
                    explanations_by_seed[seed_a],
                    explanations_by_seed[seed_b]
                )
                correlations.append(correlation)
        
        # MODIFICATA: calcola sia media che deviazione standard
        if correlations:
            mean_consistency = float(np.mean(correlations))
            std_consistency = float(np.std(correlations, ddof=1)) if len(correlations) > 1 else 0.0
        else:
            mean_consistency = 0.0
            std_consistency = 0.0
        
        if show_progress:
            print(f"  [RESULT] Consistency: {mean_consistency:.4f} ± {std_consistency:.4f}")
        
        return mean_consistency, std_consistency  # CAMBIATO: restituisce tupla
        
    finally:
        # Ripristina stato modello
        model.train(original_mode)
        for name, param in model.named_parameters():
            param.requires_grad_(original_requires_grad[name])

# MODIFICATA: Wrapper per compatibilità che restituisce tupla
def compute_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = False
) -> Tuple[float, float]:  # CAMBIATO: restituisce (media, std)
    """Wrapper per consistency che restituisce (media, std)."""
    return compute_consistency_inference_seed(
        model=model,
        tokenizer=tokenizer,
        explainer=explainer,
        texts=texts,
        seeds=seeds,
        show_progress=show_progress
    )

def evaluate_consistency_over_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = True
) -> Tuple[float, float]:  # CAMBIATO: restituisce (media, std)
    """Wrapper per consistency evaluation che restituisce (media, std)."""
    return compute_consistency_inference_seed(
        model=model,
        tokenizer=tokenizer,
        explainer=explainer,
        texts=texts,
        seeds=seeds,
        show_progress=show_progress
    )

def _compute_spearman_correlation_explanations(
    explanations_a: List[Attribution],
    explanations_b: List[Attribution]
) -> float:
    """Calcola correlazione di Spearman tra explanations."""
    correlations = []
    
    for attr_a, attr_b in zip(explanations_a, explanations_b):
        if not attr_a.tokens or not attr_b.tokens:
            continue
        
        # Trova token condivisi
        shared_scores_a, shared_scores_b = [], []
        tokens_b_set = set(attr_b.tokens)
        token_to_idx_b = {tok: i for i, tok in enumerate(attr_b.tokens)}
        
        for token, score_a in zip(attr_a.tokens, attr_a.scores):
            if token in tokens_b_set:
                idx = token_to_idx_b[token]
                shared_scores_a.append(score_a)
                shared_scores_b.append(attr_b.scores[idx])
        
        # Calcola correlazione se abbastanza token condivisi
        if len(shared_scores_a) >= MIN_SHARED_TOKENS:
            arr_a = np.array(shared_scores_a)
            arr_b = np.array(shared_scores_b)
            
            if np.var(arr_a) == 0 or np.var(arr_b) == 0:
                correlation = 1.0 if np.array_equal(arr_a, arr_b) else 0.0
            else:
                try:
                    rho, _ = spearmanr(arr_a, arr_b)
                    correlation = float(rho) if not np.isnan(rho) else 0.0
                except Exception:
                    correlation = 0.0
            
            correlations.append(correlation)
    
    return np.mean(correlations) if correlations else 0.0


# ==== 3. CONTRASTIVITY (ottimizzata) ====
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

# ==== 4. HUMAN REASONING AGREEMENT (INTEGRATO) ====
def compute_human_reasoning_score(xai_tokens: List[str], xai_scores: List[float], 
                                hr_ranking: List[str]) -> float:
    """
    Calcola Human Reasoning Agreement usando Mean Average Precision (MAP)
    
    Args:
        xai_tokens: Lista token dall'explainer
        xai_scores: Lista score corrispondenti  
        hr_ranking: Lista parole ordinate da LLM (più importante → meno importante)
    
    Returns:
        float: Average Precision score (0-1, higher is better)
    """
    if not hr_ranking or not xai_tokens or not xai_scores:
        return 0.0
    
    # Ordina XAI tokens per score (decrescente)
    xai_ranking = [token.lower() for token, score in 
                  sorted(zip(xai_tokens, xai_scores), key=lambda x: x[1], reverse=True)]
    
    # Normalizza HR ranking
    hr_set = set(word.lower() for word in hr_ranking)
    
    # Calcola Average Precision
    relevant_count = 0
    precision_sum = 0
    
    for k, xai_word in enumerate(xai_ranking, 1):
        if xai_word in hr_set:  # rel(k) = 1
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k
    
    # AP = sum(P(k) * rel(k)) / number_of_relevant_documents
    ap = precision_sum / len(hr_ranking) if hr_ranking else 0.0
    return ap

def evaluate_human_reasoning_over_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    hr_dataset,  # Pre-loaded dataset
    show_progress: bool = True
) -> float:
    """
    Valuta Human Reasoning Agreement su dataset pre-caricato
    
    Args:
        model: Modello PyTorch
        tokenizer: Tokenizer del modello
        explainer: Funzione explainer
        hr_dataset: Ground truth dataset (deve essere già caricato)
        show_progress: Mostra progress bar
    
    Returns:
        float: Mean Average Precision across dataset (0-1, higher is better)
    """
    if hr_dataset is None:
        print("[HR] No Human Reasoning dataset provided")
        return 0.0
    
    # Filtra solo esempi con HR valido
    valid_dataset = hr_dataset[hr_dataset['hr_count'] > 0].copy()
    
    if len(valid_dataset) == 0:
        print("[HR] No valid HR examples found")
        return 0.0
    
    hr_scores = []
    failed_explanations = 0
    
    iterator = tqdm(valid_dataset.iterrows(), total=len(valid_dataset), 
                   desc="HR Evaluation", leave=False) if show_progress else valid_dataset.iterrows()
    
    for idx, row in iterator:
        try:
            text = row['text']
            hr_ranking = row['hr_ranking']
            
            # Genera XAI explanation
            attr = explainer(text)
            
            # Verifica validità explanation
            if not attr.tokens or not attr.scores:
                failed_explanations += 1
                continue
            
            # Calcola HR score
            hr_score = compute_human_reasoning_score(
                attr.tokens, attr.scores, hr_ranking
            )
            hr_scores.append(hr_score)
            
        except Exception as e:
            print(f"[HR] Error processing example {idx}: {e}")
            failed_explanations += 1
            continue
    
    # Calcola MAP finale
    mean_ap = float(np.mean(hr_scores)) if hr_scores else 0.0
    
    if show_progress:
        print(f"[HR] Processed: {len(hr_scores)}/{len(valid_dataset)} examples")
        print(f"[HR] Failed explanations: {failed_explanations}")
        print(f"[HR] Mean Average Precision: {mean_ap:.4f}")
    
    return mean_ap

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

# ==== Summary Function ====
def print_metric_summary(
    robustness_score: float,
    consistency_score: float,
    contrastivity_score: float,
    human_reasoning_score: Optional[float] = None,
):
    """Stampa summary metriche."""
    print("\n" + "="*50)
    print("XAI METRICS SUMMARY")
    print("="*50)
    print(f"Robustness:    {robustness_score:.4f} (lower = more robust)")
    print(f"Consistency:   {consistency_score:.4f} (higher = more consistent)")
    print(f"Contrastivity: {contrastivity_score:.4f} (higher = more contrastive)")
    if human_reasoning_score is not None:
        print(f"Human Reasoning: {human_reasoning_score:.4f} (higher = more human-like)")
    print("="*50)

# ==== Test Function ====
if __name__ == "__main__":
    print("Testing metrics on Colab...")
    
    # Test con modello piccolo
    try:
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
        
        print("\nTesting Consistency...")
        consistency = evaluate_consistency_over_dataset(
            model, tokenizer, mock_explainer, test_texts[:2], 
            seeds=[42, 123], show_progress=False
        )
        print(f"Consistency: {consistency:.4f}")
        
        print("\nTesting Contrastivity...")
        pos_attrs = [mock_explainer(test_texts[0])]
        neg_attrs = [mock_explainer(test_texts[1])]
        contrastivity = compute_contrastivity(pos_attrs, neg_attrs)
        print(f"Contrastivity: {contrastivity:.4f}")
        
        print("\nTesting Human Reasoning...")
        # Mock HR data
        hr_ranking = ["great", "movie", "fantastic"]
        hr_score = compute_human_reasoning_score(
            ["movie", "great", "is"], [0.8, 0.9, 0.3], hr_ranking
        )
        print(f"Human Reasoning: {hr_score:.4f}")
        
        print_metric_summary(robustness, consistency, contrastivity, hr_score)
        
        print("\n Metrics test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        models.clear_gpu_memory()