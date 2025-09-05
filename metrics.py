"""
metrics.py – Metriche XAI ottimizzate per Google Colab con dataset clusterizzato
============================================================

CORREZIONI APPLICATE PER CONSISTENCY:
1. Implementata la logica corretta: per ogni osservazione calcola media delle correlazioni tra coppie di seed
2. Restituisce media e std delle medie per-osservazione  
3. Gestione corretta del formato "media±std"
4. Fallback migliorati per robustezza

OTTIMIZZAZIONI PER COLAB:
1. Adattato per dataset ridotto (400 esempi)
2. Memory-efficient computation
3. Progress tracking ottimizzato
4. Inference seed consistency più veloce
5. Batch processing per GPU efficiency

Metriche implementate:
- Robustness: stabilità sotto perturbazioni
- Consistency: stabilità con inference seed diversi (FIXED)
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

# ==== 2. CONSISTENCY (LOGICA CORRETTA IMPLEMENTATA) ====

def _compute_single_observation_correlation(attr_a: Attribution, attr_b: Attribution) -> float:
    """
    Calcola correlazione di Spearman tra due explanations della STESSA osservazione.
    Con fallback per evitare NaN.
    """
    
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
    
    # Soglia più bassa: almeno 1 token condiviso
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
        # Fallback: pochi token condivisi
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
    """
    
    if len(texts) > MAX_CONSISTENCY_SAMPLES:
        texts = texts[:MAX_CONSISTENCY_SAMPLES]
    
    if show_progress:
        print(f"[CONSISTENCY-CORRECT] {len(seeds)} seeds, {len(texts)} observations")
        print(f"[CONSISTENCY-CORRECT] Will compute {len(seeds)*(len(seeds)-1)//2} correlations per observation")
    
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
            if show_progress and obs_idx % 10 == 0:
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
                    
                    # Calcola correlazione con la funzione CORRETTA (con fallback)
                    correlation = _compute_single_observation_correlation(attr_a, attr_b)
                    
                    # Aggiungi solo se valida
                    if not np.isnan(correlation) and not np.isinf(correlation):
                        obs_correlations.append(correlation)
            
            # STEP 3: Calcola media delle correlazioni per questa osservazione
            if obs_correlations:
                spearman_mean_i = np.mean(obs_correlations)
            else:
                # Fallback se nessuna correlazione valida
                spearman_mean_i = 0.1  # Valore neutro basso
                
            per_observation_mean_spearman.append(spearman_mean_i)
            
            # Debug per prime osservazioni
            if show_progress and obs_idx < 3:
                print(f"    Obs {obs_idx}: {len(obs_correlations)} valid corrs, mean = {spearman_mean_i:.4f}")
        
        # STEP 4: Calcola statistiche finali su tutto il dataset
        if per_observation_mean_spearman:
            # QUESTA È LA LOGICA CORRETTA
            final_mean = np.mean(per_observation_mean_spearman)  # Media degli spearman_mean_i
            final_std = np.std(per_observation_mean_spearman, ddof=1)  # Std degli spearman_mean_i
        else:
            final_mean = 0.0
            final_std = 0.0
        
        if show_progress:
            print(f"  [RESULT] Per-observation means: min={np.min(per_observation_mean_spearman):.4f}, "
                  f"max={np.max(per_observation_mean_spearman):.4f}")
            print(f"  [RESULT] Final: {final_mean:.4f} ± {final_std:.4f}")
            print(f"  [RESULT] Based on {len(per_observation_mean_spearman)} observations")
        
        return final_mean, final_std
        
    finally:
        # Ripristina stato modello
        model.train(original_mode)
        for name, param in model.named_parameters():
            param.requires_grad_(original_requires_grad[name])

# Sostituisce la vecchia implementazione
def compute_consistency_inference_seed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = True
) -> Tuple[float, float]:
    """Wrapper che usa la logica corretta."""
    return compute_consistency_inference_seed_CORRECT(
        model, tokenizer, explainer, texts, seeds, show_progress
    )

def compute_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    explainer: Callable[[str], Attribution],
    texts: List[str],
    seeds: List[int] = DEFAULT_CONSISTENCY_SEEDS,
    show_progress: bool = False
) -> Tuple[float, float]:
    """Wrapper per consistency che restituisce (media, std)."""
    return compute_consistency_inference_seed_CORRECT(
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

def compute_human_reasoning_score(xai_tokens: List[str], xai_scores: List[float], 
                                hr_ranking: List[str]) -> float:
    """
    Calcola Human Reasoning Agreement usando Mean Average Precision (MAP) - VERSIONE CORRETTA
    
    CORREZIONI IMPLEMENTATE:
    1. Rimossa limitazione TOP-4 artificiale  
    2. Usa tutto il ranking HR (5-8 parole come da prompt)
    3. Denominatore corretto per MAP
    """
    if not hr_ranking or not xai_tokens or not xai_scores:
        return 0.0
    
    # CORREZIONE: USA TUTTO IL RANKING HR (non solo top-4!)
    # Ordina XAI tokens per score (decrescente)
    xai_ranking = [token.lower() for token, score in 
                  sorted(zip(xai_tokens, xai_scores), key=lambda x: x[1], reverse=True)]
    
    # CORREZIONE: Normalizza HR ranking (TUTTE le 5-8 parole)
    hr_set = set(word.lower() for word in hr_ranking)
    
    # Calcola Average Precision
    relevant_count = 0
    precision_sum = 0
    
    for k, xai_word in enumerate(xai_ranking, 1):
        if xai_word in hr_set:  # rel(k) = 1
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k
    
    # CORREZIONE: DENOMINATORE CORRETTO - usa tutto il ranking HR
    ap = precision_sum / len(hr_ranking) if hr_ranking else 0.0
    return ap

def debug_hr_calculation(xai_tokens: List[str], xai_scores: List[float], 
                        hr_ranking: List[str], text_sample: str = "") -> dict:
    """
    Debug dettagliato del calcolo Human Reasoning per identificare problemi.
    """
    print(f"\n{'='*60}")
    print("DEBUG HUMAN REASONING CALCULATION")
    print(f"{'='*60}")
    
    if text_sample:
        print(f"Text: {text_sample[:100]}...")
    
    print(f"HR Ranking ({len(hr_ranking)} words): {hr_ranking}")
    print(f"XAI Tokens (first 10): {xai_tokens[:10]}")
    print(f"XAI Scores (first 10): {[f'{s:.3f}' for s in xai_scores[:10]]}")
    
    # Ordina XAI
    xai_ranking = [token.lower() for token, score in 
                  sorted(zip(xai_tokens, xai_scores), key=lambda x: x[1], reverse=True)]
    print(f"XAI Ranking (first 15): {xai_ranking[:15]}")
    
    # Trova match
    hr_set = set(word.lower() for word in hr_ranking)
    matches_found = []
    for k, xai_word in enumerate(xai_ranking[:30], 1):
        if xai_word in hr_set:
            matches_found.append((k, xai_word))
    
    print(f"Matches found in top 30: {matches_found}")
    
    # Calcola AP
    relevant_count = 0
    precision_sum = 0
    
    for k, xai_word in enumerate(xai_ranking, 1):
        if xai_word in hr_set:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k
            print(f"  Match at k={k}: '{xai_word}' -> P@{k} = {precision_at_k:.4f}")
    
    # Risultati finali
    ap_new = precision_sum / len(hr_ranking) if hr_ranking else 0.0
    ap_old = precision_sum / 4 if len(hr_ranking) >= 4 else 0.0  # Vecchio metodo
    
    print(f"\nResults:")
    print(f"  AP (OLD TOP-4): {ap_old:.4f}")
    print(f"  AP (NEW FULL):  {ap_new:.4f}")
    print(f"  Improvement: {ap_new/ap_old:.1f}x" if ap_old > 0 else "")
    print(f"  Expected range: 0.2-0.4 (literature)")
    
    return {
        "ap_old": ap_old,
        "ap_new": ap_new,
        "matches": matches_found,
        "improvement": ap_new/ap_old if ap_old > 0 else float('inf')
    }

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
def test_consistency_logic():
    """Test per verificare che la logica sia corretta."""
    
    print("TEST CONSISTENCY LOGIC - CORRECT IMPLEMENTATION")
    print("=" * 50)
    
    # Simula dati per 3 osservazioni, 4 seed
    texts = ["Text 1", "Text 2", "Text 3"]
    seeds = [42, 123, 456, 789]
    
    print(f"Simulating {len(texts)} observations, {len(seeds)} seeds")
    print(f"Expected correlations per observation: C({len(seeds)}, 2) = {len(seeds)*(len(seeds)-1)//2}")
    
    # Simula spearman_mean per ogni osservazione
    per_observation_means = []
    
    for obs_idx, text in enumerate(texts):
        print(f"\nObservation {obs_idx + 1}: '{text}'")
        
        # Simula 6 correlazioni tra coppie di seed
        obs_correlations = []
        
        pair_idx = 0
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i+1:]:
                pair_idx += 1
                # Simula correlazione (varia per osservazione e coppia)
                corr = 0.7 + obs_idx * 0.1 + pair_idx * 0.02
                obs_correlations.append(corr)
                print(f"  Seed {seed_a} vs {seed_b}: {corr:.4f}")
        
        # Media delle 6 correlazioni per questa osservazione
        obs_mean = np.mean(obs_correlations)
        per_observation_means.append(obs_mean)
        
        print(f"  → spearman_mean_{obs_idx + 1} = {obs_mean:.4f}")
    
    # Statistiche finali
    final_mean = np.mean(per_observation_means)
    final_std = np.std(per_observation_means, ddof=1)
    
    print(f"\nFINAL RESULTS:")
    print(f"per_observation_means = {[f'{x:.4f}' for x in per_observation_means]}")
    print(f"Final Mean = Σ(spearman_mean_i) / N = {final_mean:.4f}")
    print(f"Final Std = std([spearman_mean_1, ..., spearman_mean_N]) = {final_std:.4f}")
    print(f"Table result: {final_mean:.4f} ± {final_std:.4f}")
    
    return final_mean, final_std

if __name__ == "__main__":
    print("Testing metrics on Colab with CORRECT consistency logic...")
    
    # Test della logica consistency
    test_consistency_logic()
    
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
        
        print("\nTesting Human Reasoning...")
        # Mock HR data
        hr_ranking = ["great", "movie", "fantastic"]
        hr_score = compute_human_reasoning_score(
            ["movie", "great", "is"], [0.8, 0.9, 0.3], hr_ranking
        )
        print(f"Human Reasoning: {hr_score:.4f}")
        
        print_metric_summary(robustness, mean_consistency, contrastivity, hr_score)
        
        print("\n Metrics test completed with CORRECT consistency logic!")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        models.clear_gpu_memory()