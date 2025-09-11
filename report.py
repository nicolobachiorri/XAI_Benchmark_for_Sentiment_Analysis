"""
report.py – Report XAI essenziale con --sample configurabile
====================================================================

- Solo report completi con sample size configurabile
- 3 metriche core: robustness, consistency, contrastivity
- CLI semplificata: python report.py --sample 400

Uso:
```bash
python report.py --sample 400 --models tinybert distilbert --metrics robustness consistency
python report.py --sample 200 --explainers lime shap grad_input
```
"""

import argparse
import gc
import json
import time
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

import models
import dataset
import explainers
import metrics
from utils import Timer, set_seed, aggressive_cleanup

# =============================================================================
# CONFIGURAZIONE CORE
# =============================================================================

EXPLAINERS = ["lime", "shap", "grad_input", "attention_rollout", "attention_flow", "lrp"]
METRICS = ["robustness", "contrastivity", "consistency"]
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]

RESULTS_DIR = Path("xai_results")
RESULTS_DIR.mkdir(exist_ok=True)

set_seed(42)

# =============================================================================
# BASIC MEMORY MANAGEMENT
# =============================================================================

def basic_cleanup():
    """Cleanup memoria essenziale."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_optimal_batch_size(base_batch_size: int = 10) -> int:
    """Calcola batch size ottimale basato su memoria."""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb > 8:
            return base_batch_size * 4  # 40
        elif available_gb > 4:
            return base_batch_size * 2  # 20
        elif available_gb > 2:
            return base_batch_size      # 10
        else:
            return max(2, base_batch_size // 2)  # 5
    except ImportError:
        return base_batch_size

# =============================================================================
# CORE PROCESSING FUNCTION
# =============================================================================

def process_model_core(
    model_key: str,
    explainers_to_test: List[str],
    metrics_to_compute: List[str],
    sample_size: int
) -> Dict[str, Dict[str, float]]:
    """Processa singolo modello con architettura semplificata."""
    
    print(f"\n{'='*70}")
    print(f" PROCESSING MODEL: {model_key}")
    print(f"{'='*70}")
    
    try:
        # Carica modello
        print(f"[LOAD] Loading {model_key}...")
        with Timer(f"Loading {model_key}"):
            model = models.load_model(model_key)
            tokenizer = models.load_tokenizer(model_key)
        
        # Calcola batch size
        optimal_batch_size = calculate_optimal_batch_size()
        print(f"[BATCH] Optimal batch size: {optimal_batch_size}")
        
        # Prepara dati
        print(f"[DATA] Preparing data (sample_size={sample_size})...")
        texts, labels = dataset.get_clustered_sample(sample_size, stratified=True)
        
        # Separa dati per metriche
        pos_texts = [t for t, l in zip(texts, labels) if l == 1][:optimal_batch_size]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0][:optimal_batch_size]
        consistency_texts = texts[:min(optimal_batch_size, len(texts))]
        
        print(f"[DATA] Batch sizes - Pos: {len(pos_texts)}, Neg: {len(neg_texts)}, Consistency: {len(consistency_texts)}")
        
        # Inizializza risultati
        results = {}
        for metric in metrics_to_compute:
            results[metric] = {}
            for explainer in explainers_to_test:
                if metric == "consistency":
                    results[metric][explainer] = "NaN±NaN"
                else:
                    results[metric][explainer] = float('nan')
        
        # Processing sequenziale
        print(f"\n[PROCESSING] Sequential processing of {len(explainers_to_test)} explainers...")
        
        successful_explainers = 0
        
        for i, explainer_name in enumerate(explainers_to_test, 1):
            print(f"\n[{i}/{len(explainers_to_test)}] Processing {explainer_name}...")
            print("-" * 50)
            
            explainer_success = False
            
            try:
                # Crea explainer
                explainer = explainers.get_explainer(explainer_name, model, tokenizer)
                
                # Processa ogni metrica
                for metric_name in metrics_to_compute:
                    print(f"  [METRIC] {metric_name}...", end=" ")
                    
                    try:
                        if metric_name == "robustness":
                            score = metrics.evaluate_robustness_over_dataset(
                                model, tokenizer, explainer, consistency_texts, show_progress=False
                            )
                            results[metric_name][explainer_name] = score
                            
                        elif metric_name == "contrastivity":
                            # Process in batch
                            pos_attrs = metrics.process_attributions_batch(
                                pos_texts, explainer, batch_size=optimal_batch_size//2, show_progress=False
                            )
                            neg_attrs = metrics.process_attributions_batch(
                                neg_texts, explainer, batch_size=optimal_batch_size//2, show_progress=False
                            )
                            
                            # Filter valid
                            pos_attrs = [attr for attr in pos_attrs if attr.tokens and attr.scores]
                            neg_attrs = [attr for attr in neg_attrs if attr.tokens and attr.scores]
                            
                            if pos_attrs and neg_attrs:
                                score = metrics.compute_contrastivity(pos_attrs, neg_attrs)
                            else:
                                score = 0.0
                            
                            results[metric_name][explainer_name] = score
                                
                        elif metric_name == "consistency":
                            # Usa la logica corretta
                            mean_score, std_score = metrics.evaluate_consistency_over_dataset(
                                model=model,
                                tokenizer=tokenizer,
                                explainer=explainer,
                                texts=consistency_texts,
                                seeds=DEFAULT_CONSISTENCY_SEEDS,
                                show_progress=False
                            )
                            
                            # Crea stringa formattata
                            formatted_score = f"{mean_score:.4f}±{std_score:.4f}"
                            results[metric_name][explainer_name] = formatted_score
                            score = mean_score
                        
                        print(f" SUCCESS {score:.4f}")
                        explainer_success = True
                        
                    except Exception as e:
                        print(f"ERROR: {str(e)[:50]}...")
                        if metric_name == "consistency":
                            results[metric_name][explainer_name] = "NaN±NaN"
                        else:
                            results[metric_name][explainer_name] = float('nan')
                
                if explainer_success:
                    successful_explainers += 1
                
                # Cleanup explainer
                del explainer
                basic_cleanup()
                
            except Exception as e:
                print(f"  EXPLAINER FAILED: {explainer_name}: {e}")
                for metric in metrics_to_compute:
                    if metric == "consistency":
                        results[metric][explainer_name] = "NaN±NaN"
                    else:
                        results[metric][explainer_name] = float('nan')
        
        print(f"\n[COMPLETE] {model_key} processing completed")
        print(f"[STATS] Successful explainers: {successful_explainers}/{len(explainers_to_test)}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Model {model_key} processing failed: {e}")
        
        # Restituisci struttura vuota ma valida
        empty_results = {}
        for metric in metrics_to_compute:
            empty_results[metric] = {}
            for explainer in explainers_to_test:
                if metric == "consistency":
                    empty_results[metric][explainer] = "NaN±NaN"
                else:
                    empty_results[metric][explainer] = float('nan')
        
        return empty_results
        
    finally:
        # Cleanup finale
        print(f"[CLEANUP] Final cleanup for {model_key}...")
        aggressive_cleanup()

# =============================================================================
# TABLE BUILDING
# =============================================================================

def build_report_tables(all_results: Dict[str, Dict], metrics_to_compute: List[str]) -> Dict[str, pd.DataFrame]:
    """Costruisce tabelle finali dai risultati."""
    print(f"\n{'='*70}")
    print(" BUILDING REPORT TABLES")
    print(f"{'='*70}")
    
    tables = {}
    
    for metric in metrics_to_compute:
        print(f"\n[TABLE] Building {metric} table...")
        
        metric_data = defaultdict(dict)
        
        for model_key, model_data in all_results.items():
            if "results" not in model_data:
                continue
                
            if metric not in model_data["results"]:
                continue
                
            explainer_scores = model_data["results"][metric]
            
            for explainer_name, score in explainer_scores.items():
                if score is not None:
                    if metric == "consistency" and isinstance(score, str) and "±" in score:
                        # Mantieni formato stringa per consistency
                        metric_data[explainer_name][model_key] = score
                    elif not (isinstance(score, float) and np.isnan(score)):
                        metric_data[explainer_name][model_key] = score
        
        if metric_data:
            df = pd.DataFrame(metric_data).T
            tables[metric] = df
            print(f"[TABLE] {metric}: {df.shape[0]} explainers × {df.shape[1]} models")
        else:
            print(f"[TABLE] {metric}: No data available")
            tables[metric] = pd.DataFrame()
    
    return tables

def print_table_analysis(df: pd.DataFrame, metric_name: str):
    """Analisi e interpretazione tabella."""
    print(f"\n{'='*60}")
    print(f" {metric_name.upper()} ANALYSIS")
    print(f"{'='*60}")
    
    if df.empty:
        print("No data available for analysis")
        return
    
    if metric_name == "consistency":
        print(" Per-Explainer Statistics (mean ± std):")
        print("-" * 50)
        
        for explainer in df.index:
            values_str = df.loc[explainer].dropna()
            if len(values_str) > 0:
                means = []
                stds = []
                for val_str in values_str:
                    if isinstance(val_str, str) and "±" in val_str:
                        try:
                            mean_part, std_part = val_str.split("±")
                            means.append(float(mean_part))
                            stds.append(float(std_part))
                        except ValueError:
                            continue
                
                if means:
                    avg_mean = np.mean(means)
                    avg_std = np.mean(stds)
                    count = len(means)
                    coverage = count / len(df.columns)
                    
                    min_mean = np.min(means)
                    max_mean = np.max(means)
                    
                    print(f"  {explainer:>15s}: μ={avg_mean:.4f}±{avg_std:.4f} "
                          f"range=[{min_mean:.4f},{max_mean:.4f}] (n={count}, {coverage:.1%})")
        
        return
    
    # Analisi per altri metrics
    print(" Per-Explainer Statistics:")
    print("-" * 40)
    for explainer in df.index:
        values = df.loc[explainer].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            count = len(values)
            coverage = len(values) / len(df.columns)
            print(f"  {explainer:>15s}: μ={mean_val:.4f} σ={std_val:.4f} (n={count}, {coverage:.1%} coverage)")
    
    print("\n Per-Model Statistics:")
    print("-" * 40)
    for model in df.columns:
        values = df[model].dropna()
        if len(values) > 0:
            mean_val = values.mean()
            std_val = values.std()
            count = len(values)
            coverage = len(values) / len(df.index)
            print(f"  {model:>15s}: μ={mean_val:.4f} σ={std_val:.4f} (n={count}, {coverage:.1%} coverage)")

# =============================================================================
# MAIN REPORT FUNCTION
# =============================================================================

def run_complete_report(
    models_to_test: List[str] = None,
    explainers_to_test: List[str] = None, 
    metrics_to_compute: List[str] = None,
    sample_size: int = 100,
    resume: bool = True
) -> Dict[str, pd.DataFrame]:
    """Esegue report completo XAI."""
    
    start_time = time.time()
    
    try:
        print("="*80)
        print(" XAI COMPLETE REPORT (SIMPLIFIED)")
        print("="*80)
        print(" 3 Metrics: Robustness, Consistency, Contrastivity")
        print(" Consistency: CORRECT LOGIC - per-observation mean correlations")
        print("="*80)
        
        # Setup defaults
        available_models, available_explainers = get_available_resources()
        
        if models_to_test is None:
            models_to_test = available_models
        if explainers_to_test is None:
            explainers_to_test = available_explainers
        if metrics_to_compute is None:
            metrics_to_compute = METRICS
        
        # Filter available
        models_to_test = [m for m in models_to_test if m in available_models]
        explainers_to_test = [e for e in explainers_to_test if e in available_explainers]
        
        total_combinations = len(models_to_test) * len(explainers_to_test) * len(metrics_to_compute)
        
        print(f"\n[REPORT] Configuration:")
        print(f"  Models: {models_to_test}")
        print(f"  Explainers: {explainers_to_test}")
        print(f"  Metrics: {metrics_to_compute}")
        print(f"  Sample size: {sample_size}")
        print(f"  Total combinations: {total_combinations}")
        
        # Process each model
        print(f"\n{'='*80}")
        print("MODEL PROCESSING")
        print(f"{'='*80}")
        
        all_results = {}
        
        for i, model_key in enumerate(models_to_test, 1):
            print(f"\n[{i}/{len(models_to_test)}] Model: {model_key}")
            
            try:
                with Timer(f"Processing {model_key}"):
                    results = process_model_core(
                        model_key=model_key,
                        explainers_to_test=explainers_to_test,
                        metrics_to_compute=metrics_to_compute,
                        sample_size=sample_size
                    )
                    all_results[model_key] = {"results": results, "completed": True}
                
                print(f"[SUCCESS] {model_key} completed successfully")
                
            except Exception as e:
                print(f"[ERROR] Model {model_key} failed: {e}")
                all_results[model_key] = {
                    "results": {metric: {} for metric in metrics_to_compute},
                    "completed": False,
                    "error": str(e)
                }
        
        # Build tables
        print(f"\n{'='*80}")
        print("BUILDING TABLES")
        print(f"{'='*80}")
        
        tables = build_report_tables(all_results, metrics_to_compute)
        
        # Analysis & Output
        print(f"\n{'='*80}")
        print("ANALYSIS & OUTPUT")
        print(f"{'='*80}")
        
        execution_time = time.time() - start_time
        
        # Print tables with analysis
        for metric_name, df in tables.items():
            if not df.empty:
                print(f"\n {metric_name.upper()} TABLE:")
                print("=" * 50)
                
                if metric_name == "consistency":
                    print(df.to_string(na_rep="—"))
                else:
                    print(df.to_string(float_format="%.4f", na_rep="—"))
                
                # Save CSV
                csv_file = RESULTS_DIR / f"{metric_name}_table_complete.csv"
                df.to_csv(csv_file)
                print(f"[SAVE] CSV saved: {csv_file}")
                
                # Analysis
                print_table_analysis(df, metric_name)
        
        # Final summary
        print(f"\n{'='*80}")
        print(" COMPLETE REPORT FINISHED!")
        print(f"{'='*80}")
        print(f"  Total time: {execution_time/60:.1f} minutes")
        print(f"  Models processed: {len([r for r in all_results.values() if r.get('completed', False)])}/{len(models_to_test)}")
        print(f"  Tables generated: {len([t for t in tables.values() if not t.empty])}")
        print(f"  Files saved in: {RESULTS_DIR}")
        
        return tables
        
    except Exception as e:
        print(f"\nCOMPLETE REPORT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_available_resources():
    """Ottieni risorse disponibili."""
    available_models = list(models.MODELS.keys())
    available_explainers = [exp for exp in EXPLAINERS if exp in explainers.list_explainers()]
    
    print(f"[RESOURCES] Models: {len(available_models)} available")
    print(f"[RESOURCES] Explainers: {len(available_explainers)} available")
    print(f"[RESOURCES] Metrics: {len(METRICS)} available")
    print(f"[RESOURCES] Dataset: {len(dataset.test_df)} clustered examples")
    
    return available_models, available_explainers

# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="XAI Complete Report Generator")
    parser.add_argument("--sample", type=int, default=100, help="Sample size (default: 100)")
    parser.add_argument("--models", nargs="+", choices=list(models.MODELS.keys()), 
                       default=None, help="Models to test")
    parser.add_argument("--explainers", nargs="+", choices=EXPLAINERS,
                       default=None, help="Explainers to test")
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                       default=None, help="Metrics to compute")
    parser.add_argument("--no-resume", dest="resume", action="store_false", 
                       default=True, help="Start fresh (no resume)")
    parser.add_argument("--no-backup", dest="backup_to_drive", action="store_false",
                       default=True, help="Skip Google Drive backup")
    
    args = parser.parse_args()
    
    print("XAI COMPLETE REPORT - SIMPLIFIED VERSION")
    print("="*50)
    print("3 Core Metrics: Robustness, Consistency, Contrastivity")
    print("Consistency: CORRECT LOGIC implementation")
    if args.backup_to_drive:
        print("Google Drive backup: ENABLED")
    else:
        print("Google Drive backup: DISABLED")
    print("="*50)
    
    tables = run_complete_report(
        models_to_test=args.models,
        explainers_to_test=args.explainers,
        metrics_to_compute=args.metrics,
        sample_size=args.sample,
        resume=args.resume,
        backup_to_drive=args.backup_to_drive
    )
    
    if not any(not df.empty for df in tables.values()):
        print("No results generated!")
        sys.exit(1)
    else:
        print("Complete report generated successfully!")

if __name__ == "__main__":
    main()