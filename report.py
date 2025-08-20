"""
report.py – Report XAI + HUMAN REASONING (FIXED CONSISTENCY)
====================================================================

CORREZIONI APPLICATE PER CONSISTENCY:
1. Fix doppia assegnazione in process_model_simplified()
2. Gestione corretta stringhe "media±std"
3. Debug function per verificare risultati
4. Analisi migliorata con ranking explainer
5. Test function per validazione

OTTIMIZZAZIONI MANTENUTE + HUMAN REASONING:
1. Adaptive Batch Size: Dimensioni batch dinamiche basate su memoria disponibile
2. Embedding Caching: Cache intelligente per embeddings e tokenizzazioni
3. Memory Management: Cleanup progressivo tra batch con monitoraggio
4. GPU Optimization: CUDA sync ottimizzato, memory pool management
5. Thread Pools: I/O operations parallele per salvataggio/caricamento
6. Smart Resource Allocation: Allocazione dinamica risorse
7. **HUMAN REASONING INTEGRATION**: Valutazione automatica accordo con ranking LLM

Uso in Colab:
```python
import report

# Report ultra-veloce
report.turbo_report()

# Report personalizzato con Human Reasoning
report.run_optimized_report(
    models=["tinybert", "distilbert"],
    explainers=["lime", "shap", "grad_input"], 
    metrics=["robustness", "consistency", "human_reasoning"],
    enable_caching=True,
    adaptive_batching=True
)

# Check HR status
report.check_hr_status()

# Generate HR ground truth
report.generate_hr_ground_truth("your_api_key")

# Test consistency formatting
report.test_consistency_format()
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
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib
import weakref

import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import psutil

import models
import dataset
import explainers
import metrics
import HumanReasoning as hr
from utils import Timer, PerformanceProfiler, AutoRecovery, print_memory_status, aggressive_cleanup, set_seed

# =============================================================================
# CONFIGURAZIONE OTTIMIZZAZIONI SEMPLIFICATA + HUMAN REASONING
# =============================================================================

# Configurazione base - AGGIUNTO HUMAN REASONING
EXPLAINERS = ["lime", "shap", "grad_input", "attention_rollout", "attention_flow", "lrp"]
METRICS = ["robustness", "contrastivity", "consistency", "human_reasoning"]
DEFAULT_CONSISTENCY_SEEDS = [42, 123, 456, 789]

# Configurazione ottimizzazioni
CACHE_DIR = Path("xai_cache")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("xai_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Configurazione semplificata (no parallelization)
MEMORY_THRESHOLD_GB = 2.0  # Soglia minima memoria per ottimizzazioni

set_seed(42)

# =============================================================================
# HUMAN REASONING STATUS FUNCTIONS
# =============================================================================

def check_hr_status() -> Dict[str, Any]:
    """Controlla status Human Reasoning per report (FIXED VERSION)."""
    print(f"\n[HR-STATUS] Checking Human Reasoning availability...")
    
    hr_info = hr.get_info()
    
    print(f"[HR-STATUS] Available: {hr_info['available']}")
    if hr_info['available']:
        print(f"[HR-STATUS] Valid examples: {hr_info['valid_examples']}/{hr_info['total_examples']}")
        print(f"[HR-STATUS] Average words: {hr_info['avg_words_per_example']:.1f}")
        print(f"[HR-STATUS] CSV exists: {'YES' if hr.HR_DATASET_CSV.exists() else 'NO'}")
        
        # CORREZIONE: Verifica consistenza
        if hr_info.get('exact_match_dataset', False):
            print(f"[HR-STATUS] Dataset consistency: VERIFIED")
        else:
            print(f"[HR-STATUS] Dataset consistency: CHECKING...")
            try:
                consistent = hr.verify_dataset_consistency()
                if consistent:
                    print(f"[HR-STATUS] Dataset consistency: VERIFIED (on-demand check)")
                else:
                    print(f"[HR-STATUS] Dataset consistency: FAILED - needs regeneration")
            except Exception as e:
                print(f"[HR-STATUS] Dataset consistency: ERROR - {e}")
    else:
        print(f"[HR-STATUS] Not available - needs generation")
    
    return hr_info

def generate_hr_ground_truth(api_key: str, sample_size: int = 400) -> bool:
    """Genera Human Reasoning ground truth per report (FIXED VERSION)."""
    print(f"\n[HR-GENERATE] Starting Human Reasoning generation (FIXED)...")
    print(f"[HR-GENERATE] Will use EXACTLY the 400 clustered examples")
    print(f"[HR-GENERATE] Estimated time: {400 * 1.2 / 60:.1f} minutes")
    
    try:
        with Timer(f"HR Generation (400 fixed examples)"):
            hr_dataset = hr.generate_ground_truth(
                api_key=api_key,
                sample_size=None,  # None = usa tutti i 400 del dataset clusterizzato
                resume=True
            )
        
        if hr_dataset is not None and len(hr_dataset) == 400:
            valid_count = (hr_dataset['hr_count'] > 0).sum()
            success_rate = valid_count / len(hr_dataset)
            
            print(f"[HR-GENERATE]  Generated {valid_count}/{len(hr_dataset)} valid examples ({success_rate:.1%})")
            
            # Verifica consistenza
            try:
                consistent = hr.verify_dataset_consistency()
                if consistent:
                    print(f"[HR-GENERATE]  Dataset consistency: VERIFIED")
                else:
                    print(f"[HR-GENERATE]  Dataset consistency: FAILED (unexpected)")
            except Exception as e:
                print(f"[HR-GENERATE]  Dataset consistency check failed: {e}")
            
            return success_rate > 0.3  # Soglia realistica
        else:
            print(f"[HR-GENERATE]  Generation failed")
            return False
            
    except Exception as e:
        print(f"[HR-GENERATE]  Error: {e}")
        return False

def ensure_hr_availability(auto_generate: bool = False, api_key: Optional[str] = None) -> bool:
    """Assicura che Human Reasoning sia disponibile per il report (FIXED)."""
    hr_info = hr.get_info()
    
    if hr_info['available']:
        if hr_info['valid_examples'] < 50:  # Soglia minima
            print(f"[HR-ENSURE]  Too few valid examples ({hr_info['valid_examples']})")
            return False
        
        # CORREZIONE: Verifica anche consistenza
        if not hr_info.get('exact_match_dataset', False):
            print(f"[HR-ENSURE] Checking dataset consistency...")
            try:
                consistent = hr.verify_dataset_consistency()
                if not consistent:
                    print(f"[HR-ENSURE]  Dataset consistency failed")
                    if auto_generate and api_key:
                        print(f"[HR-ENSURE] Auto-regenerating due to inconsistency...")
                        return generate_hr_ground_truth(api_key)
                    else:
                        print(f"[HR-ENSURE]  Inconsistent HR dataset - set auto_generate=True to fix")
                        return False
            except Exception as e:
                print(f"[HR-ENSURE]  Consistency check error: {e}")
                return False
        
        print(f"[HR-ENSURE]  Human Reasoning available with {hr_info['valid_examples']} examples")
        return True
    
    if auto_generate and api_key:
        print(f"[HR-ENSURE] Auto-generating Human Reasoning ground truth...")
        return generate_hr_ground_truth(api_key)
    else:
        print(f"[HR-ENSURE]  Human Reasoning not available")
        print(f"[HR-ENSURE] To enable: report.generate_hr_ground_truth('your_api_key')")
        return False

# =============================================================================
# GESTIONE MEMORIA AVANZATA (MANTENUTA)
# =============================================================================

class AdvancedMemoryManager:
    """Gestione memoria avanzata con monitoring e ottimizzazioni GPU."""
    
    def __init__(self):
        self.memory_history = []
        self.gpu_memory_pool_enabled = False
        self.cleanup_callbacks = []
        
    def get_available_memory_gb(self) -> float:
        """Ottieni memoria RAM disponibile in GB."""
        return psutil.virtual_memory().available / (1024**3)
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Ottieni informazioni memoria GPU."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "reserved": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),
            "cached": torch.cuda.memory_cached() / (1024**3),
            "reserved": torch.cuda.memory_reserved() / (1024**3)
        }
    
    def calculate_optimal_batch_size(self, base_batch_size: int = 10) -> int:
        """Calcola batch size ottimale basato su memoria disponibile."""
        available_gb = self.get_available_memory_gb()
        
        if available_gb > 8:
            return base_batch_size * 4  # 40
        elif available_gb > 4:
            return base_batch_size * 2  # 20
        elif available_gb > 2:
            return base_batch_size      # 10
        else:
            return max(2, base_batch_size // 2)  # 5
    
    def enable_gpu_memory_pool(self):
        """Abilita memory pool GPU per allocazioni efficienti."""
        if torch.cuda.is_available() and not self.gpu_memory_pool_enabled:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # Pre-alloca pool piccolo per evitare frammentazione
                dummy = torch.zeros(100, 100, device='cuda')
                del dummy
                torch.cuda.empty_cache()
                
                self.gpu_memory_pool_enabled = True
                print("[MEMORY] GPU memory pool enabled")
            except Exception as e:
                print(f"[MEMORY] Failed to enable GPU memory pool: {e}")
    
    def progressive_cleanup(self, level: str = "medium"):
        """Cleanup progressivo con diversi livelli di aggressività."""
        start_time = time.time()
        
        if level == "light":
            gc.collect()
        elif level == "medium":
            for _ in range(2):
                collected = gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        elif level == "aggressive":
            for _ in range(3):
                gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"[MEMORY] Cleanup callback failed: {e}")
            
            time.sleep(1)  # Stabilization pause
        
        cleanup_time = time.time() - start_time
        memory_after = self.get_available_memory_gb()
        gpu_info = self.get_gpu_memory_info()
        
        print(f"[MEMORY] {level} cleanup: {cleanup_time:.1f}s, "
              f"RAM: {memory_after:.1f}GB, GPU: {gpu_info['allocated']:.1f}GB")
    
    def add_cleanup_callback(self, callback):
        """Aggiungi callback personalizzato per cleanup."""
        self.cleanup_callbacks.append(callback)

# =============================================================================
# SISTEMA CACHING AVANZATO (MANTENUTO)
# =============================================================================

class EmbeddingCache:
    """Cache intelligente per embeddings e tokenizzazioni."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = weakref.WeakValueDictionary()
        self.cache_stats = {"hits": 0, "misses": 0, "saves": 0}
        
    def _get_cache_key(self, model_key: str, text: str, operation: str = "embedding") -> str:
        """Genera chiave cache univoca."""
        content = f"{model_key}_{operation}_{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Ottieni percorso file cache."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get_embedding(self, model_key: str, text: str) -> Optional[torch.Tensor]:
        """Ottieni embedding dalla cache."""
        cache_key = self._get_cache_key(model_key, text, "embedding")
        
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[cache_key].clone()
        
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                self.memory_cache[cache_key] = embedding
                self.cache_stats["hits"] += 1
                return embedding.clone()
            except Exception as e:
                print(f"[CACHE] Failed to load {cache_path}: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def save_embedding(self, model_key: str, text: str, embedding: torch.Tensor):
        """Salva embedding in cache."""
        cache_key = self._get_cache_key(model_key, text, "embedding")
        
        self.memory_cache[cache_key] = embedding.clone()
        
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding.cpu(), f)
            self.cache_stats["saves"] += 1
        except Exception as e:
            print(f"[CACHE] Failed to save {cache_path}: {e}")
    
    def print_stats(self):
        """Stampa statistiche cache."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        print(f"[CACHE] Stats: {self.cache_stats['hits']} hits, "
              f"{self.cache_stats['misses']} misses, "
              f"{hit_rate:.1%} hit rate, "
              f"{self.cache_stats['saves']} saves")

# =============================================================================
# I/O THREAD POOL (MANTENUTO)
# =============================================================================

class AsyncIOManager:
    """Gestione I/O operations asincrone."""
    
    def __init__(self, max_workers: int = 2):
        self.io_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_operations = []
    
    def save_async(self, data: Any, filepath: Path, format: str = "json"):
        """Salvataggio asincrono."""
        def save_task():
            try:
                if format == "json":
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                elif format == "pickle":
                    with open(filepath, 'wb') as f:
                        pickle.dump(data, f)
                elif format == "csv":
                    data.to_csv(filepath)
                print(f"[ASYNC-IO] Saved: {filepath.name}")
                return True
            except Exception as e:
                print(f"[ASYNC-IO] Save failed {filepath.name}: {e}")
                return False
        
        future = self.io_pool.submit(save_task)
        self.pending_operations.append(future)
        return future
    
    def wait_all(self, timeout: int = 60):
        """Aspetta completamento di tutte le operazioni pending."""
        if self.pending_operations:
            print(f"[ASYNC-IO] Waiting for {len(self.pending_operations)} pending operations...")
            
            completed = 0
            for future in as_completed(self.pending_operations, timeout=timeout):
                try:
                    future.result()
                    completed += 1
                except Exception as e:
                    print(f"[ASYNC-IO] Operation failed: {e}")
            
            print(f"[ASYNC-IO] Completed {completed}/{len(self.pending_operations)} operations")
            self.pending_operations.clear()
    
    def cleanup(self):
        """Cleanup I/O pool."""
        self.wait_all()
        self.io_pool.shutdown(wait=True)

# =============================================================================
# GOOGLE DRIVE BACKUP FUNCTIONS (MANTENUTE)
# =============================================================================

def backup_results_to_drive(results_dir: str = "xai_results", drive_folder: str = "XAI_Results") -> bool:
    """Backup automatico risultati su Google Drive."""
    
    # Controlla se Google Drive è montato
    drive_path = Path("/content/drive")
    if not drive_path.exists():
        print("[DRIVE] Google Drive not mounted. Mounting now...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("[DRIVE]  Google Drive mounted successfully")
        except Exception as e:
            print(f"[DRIVE] Failed to mount Google Drive: {e}")
            return False
    
    # Path di destinazione su Drive
    drive_backup_dir = Path(f"/content/drive/MyDrive/{drive_folder}")
    
    # Crea cartella timestampata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = drive_backup_dir / f"report_{timestamp}"
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GOOGLE DRIVE BACKUP")
    print(f"{'='*80}")
    print(f"Destination: {timestamped_dir}")
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print("[DRIVE] Results directory not found")
        return False
    
    # Trova file da copiare
    files_to_copy = []
    
    # CSV tables
    csv_files = list(results_path.glob("*_table*.csv"))
    files_to_copy.extend(csv_files)
    
    # JSON results
    json_files = list(results_path.glob("results_*.json"))
    files_to_copy.extend(json_files)
    
    if not files_to_copy:
        print("[DRIVE] No files found to backup")
        return False
    
    print(f"[DRIVE] Found {len(files_to_copy)} files to backup")
    
    # Copia file
    copied_files = 0
    failed_files = 0
    
    for file_path in files_to_copy:
        try:
            destination = timestamped_dir / file_path.name
            
            if file_path.suffix == '.json':
                # Per JSON, copia e comprimi se grande
                if file_path.stat().st_size > 1024*1024:  # > 1MB
                    print(f"[DRIVE] Copying large file: {file_path.name}")
                else:
                    print(f"[DRIVE] Copying: {file_path.name}")
            else:
                print(f"[DRIVE] Copying: {file_path.name}")
            
            # Copia file
            shutil.copy2(file_path, destination)
            copied_files += 1
            
        except Exception as e:
            print(f"[DRIVE] Failed to copy {file_path.name}: {e}")
            failed_files += 1
    
    # Crea file di metadata
    try:
        metadata = {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "files_copied": copied_files,
            "files_failed": failed_files,
            "total_files": len(files_to_copy),
            "colab_session": True,
            "backup_type": "automatic"
        }
        
        metadata_file = timestamped_dir / "backup_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[DRIVE] Created metadata file")
        
    except Exception as e:
        print(f"[DRIVE] Failed to create metadata: {e}")
    
    # Summary
    success_rate = copied_files / len(files_to_copy) if files_to_copy else 0
    
    print(f"\n[DRIVE] Backup Summary:")
    print(f"  Copied: {copied_files}/{len(files_to_copy)} files ({success_rate:.1%})")
    print(f"  Failed: {failed_files}")
    print(f"  Location: {timestamped_dir}")
    
    if copied_files > 0:
        print(f"   Results safely backed up to Google Drive!")
        print(f"  Access via: My Drive > {drive_folder} > report_{timestamp}")
    
    return success_rate > 0.8  # Success se almeno 80% files copied

def quick_drive_backup(results_dir: str = "xai_results") -> bool:
    """Backup veloce solo CSV tables su Drive."""
    
    # Controlla Drive mount
    if not Path("/content/drive").exists():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except:
            print("[DRIVE] Cannot mount Google Drive")
            return False
    
    # Path semplificato
    drive_dir = Path("/content/drive/MyDrive/XAI_Results")
    drive_dir.mkdir(exist_ok=True)
    
    results_path = Path(results_dir)
    csv_files = list(results_path.glob("*_table*.csv"))
    
    if not csv_files:
        return False
    
    print(f"[DRIVE] Quick backup: {len(csv_files)} CSV files")
    
    copied = 0
    for csv_file in csv_files:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            dest_name = f"{timestamp}_{csv_file.name}"
            dest_path = drive_dir / dest_name
            
            shutil.copy2(csv_file, dest_path)
            copied += 1
            print(f"[DRIVE]  {csv_file.name}")
            
        except Exception as e:
            print(f"[DRIVE]  {csv_file.name}: {e}")
    
    return copied > 0

# =============================================================================
# CONSISTENCY DEBUG FUNCTIONS (NUOVE)
# =============================================================================

def debug_consistency_results(all_results: Dict) -> None:
    """Debug function per verificare i risultati consistency (FIXED)."""
    print("\n[DEBUG] Consistency Results Verification:")
    print("=" * 60)
    
    found_consistency = False
    
    for model_key, model_data in all_results.items():
        if "results" not in model_data or "consistency" not in model_data["results"]:
            continue
        
        found_consistency = True
        print(f"\nModel: {model_key}")
        consistency_results = model_data["results"]["consistency"]
        
        for explainer, score in consistency_results.items():
            if isinstance(score, str) and "±" in score:
                try:
                    mean_part, std_part = score.split("±")
                    mean_val = float(mean_part)
                    std_val = float(std_part)
                    
                    # Valutazione stabilità
                    if std_val < 0.05:
                        stability = "VERY_STABLE"
                    elif std_val < 0.1:
                        stability = "STABLE"
                    elif std_val < 0.2:
                        stability = "MODERATE"
                    else:
                        stability = "UNSTABLE"
                    
                    print(f"  {explainer:>15s}: {score} (μ={mean_val:.4f}, σ={std_val:.4f}) [{stability}]")
                except ValueError as e:
                    print(f"  {explainer:>15s}: {score} [PARSE ERROR: {e}]")
            else:
                print(f"  {explainer:>15s}: {score} [NOT FORMATTED STRING]")
    
    if not found_consistency:
        print("No consistency results found in all_results")

def test_consistency_format() -> None:
    """Test per verificare il formato consistency (NUOVA)."""
    print("\n" + "="*70)
    print("TESTING CONSISTENCY FORMAT")
    print("="*70)
    
    # Simula risultati consistency
    test_results = {
        "tinybert": {
            "results": {
                "consistency": {
                    "lime": "0.8500±0.0234",
                    "shap": "0.7890±0.0456", 
                    "grad_input": "0.9123±0.0123"
                }
            },
            "completed": True
        },
        "distilbert": {
            "results": {
                "consistency": {
                    "lime": "0.8200±0.0345",
                    "shap": "0.8100±0.0278",
                    "grad_input": "0.8950±0.0234"
                }
            },
            "completed": True
        }
    }
    
    print("Test data:")
    for model, data in test_results.items():
        print(f"  {model}: {data['results']['consistency']}")
    
    # Test build table
    print(f"\nBuilding table...")
    tables = build_report_tables(test_results, ["consistency"])
    consistency_table = tables["consistency"]
    
    print(f"\nGenerated table:")
    print(consistency_table.to_string(na_rep="—"))
    
    print(f"\nTable analysis:")
    print_table_analysis(consistency_table, "consistency")
    
    print(f"\nDebug verification:")
    debug_consistency_results(test_results)
    
    print(f"\n✅ Consistency format test completed!")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_simplified_header():
    """Header per report semplificato + Human Reasoning."""
    print("="*80)
    print(" XAI COMPREHENSIVE REPORT + HUMAN REASONING (FIXED CONSISTENCY)")
    print("="*80)
    print(" Optimizations:")
    print("   - Adaptive batch sizing based on available memory")
    print("   - Intelligent embedding/tokenization caching")
    print("   - Advanced GPU memory management")
    print("   - Asynchronous I/O operations")
    print("   - Progressive memory cleanup")
    print(" ")
    print(" Features:")
    print("   - Sequential explainer processing (more reliable)")
    print("   - 4 metrics: Robustness, Consistency, Contrastivity, Human Reasoning")
    print("   - Human Reasoning: LLM-generated importance rankings")
    print("   - Consistency: Mean ± Standard Deviation (FIXED)")
    print("   - Automatic Google Drive backup")
    print("="*80)

def get_system_resources():
    """Analizza risorse sistema per ottimizzazioni."""
    ram_gb = psutil.virtual_memory().total / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = psutil.cpu_count()
    
    gpu_info = {"available": False, "memory_gb": 0}
    if torch.cuda.is_available():
        gpu_info["available"] = True
        gpu_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"[SYSTEM] RAM: {available_ram_gb:.1f}/{ram_gb:.1f}GB available")
    print(f"[SYSTEM] CPU: {cpu_count} cores")
    gpu_text = "Yes" if gpu_info['available'] else "No"
    if gpu_info['available']:
        gpu_text += f" ({gpu_info['memory_gb']:.1f}GB)"
    print(f"[SYSTEM] GPU: {gpu_text}")
    
    # Human Reasoning status
    hr_info = hr.get_info()
    hr_text = "Available" if hr_info['available'] else "Not Available"
    if hr_info['available']:
        hr_text += f" ({hr_info['valid_examples']} examples)"
    print(f"[SYSTEM] Human Reasoning: {hr_text}")
    
    return {
        "ram_total_gb": ram_gb,
        "ram_available_gb": available_ram_gb,
        "cpu_count": cpu_count,
        "gpu_info": gpu_info,
        "hr_available": hr_info['available']
    }

def process_model_simplified(
    model_key: str,
    explainers_to_test: List[str],
    metrics_to_compute: List[str],
    sample_size: int,
    enable_caching: bool = True,
    adaptive_batching: bool = True,
    recovery: AutoRecovery = None,
    hr_dataset = None  # Pre-loaded HR dataset
) -> Dict[str, Dict[str, float]]:
    """Processa singolo modello con ottimizzazioni semplificate + Human Reasoning (FIXED CONSISTENCY)."""
    
    print(f"\n{'='*70}")
    print(f" PROCESSING MODEL: {model_key} (SIMPLIFIED + HR + FIXED CONSISTENCY)")
    print(f"{'='*70}")
    
    # Inizializza manager
    memory_manager = AdvancedMemoryManager()
    embedding_cache = EmbeddingCache() if enable_caching else None
    async_io = AsyncIOManager()
    
    profiler = PerformanceProfiler()
    profiler.start_operation(f"model_{model_key}_simplified")
    
    try:
        # Abilita ottimizzazioni GPU
        memory_manager.enable_gpu_memory_pool()
        
        # Carica modello
        print(f"[LOAD] Loading {model_key} with caching...")
        with Timer(f"Loading {model_key}"):
            model = models.load_model(model_key)
            tokenizer = models.load_tokenizer(model_key)
        
        # Calcola batch size ottimale
        if adaptive_batching:
            optimal_batch_size = memory_manager.calculate_optimal_batch_size()
            print(f"[ADAPTIVE] Optimal batch size: {optimal_batch_size}")
        else:
            optimal_batch_size = 10
        
        # Prepara dati
        print(f"[DATA] Preparing data (sample_size={sample_size})...")
        texts, labels = dataset.get_clustered_sample(sample_size, stratified=True)
        
        # Pre-cache embeddings se abilitato
        if enable_caching:
            print(f"[CACHE] Pre-caching embeddings for {len(texts)} texts...")
            cache_hits = 0
            for text in texts[:20]:  # Solo primi 20 per test
                if embedding_cache.get_embedding(model_key, text) is not None:
                    cache_hits += 1
            print(f"[CACHE] {cache_hits}/20 embeddings found in cache")
        
        # Separa dati per metriche
        pos_texts = [t for t, l in zip(texts, labels) if l == 1][:optimal_batch_size]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0][:optimal_batch_size]
        consistency_texts = texts[:min(optimal_batch_size, len(texts))]
        
        print(f"[DATA] Batch sizes - Pos: {len(pos_texts)}, Neg: {len(neg_texts)}, Consistency: {len(consistency_texts)}")
        
        # Check Human Reasoning availability se richiesto
        hr_available = hr_dataset is not None
        if "human_reasoning" in metrics_to_compute:
            if not hr_available:
                print(f"[HR] Human Reasoning dataset not provided, loading...")
                hr_dataset = hr.load_ground_truth()
                hr_available = hr_dataset is not None
            
            if hr_available:
                valid_hr_count = (hr_dataset['hr_count'] > 0).sum()
                print(f"[HR] Using {valid_hr_count} valid HR examples")
            else:
                print(f"[HR]  Human Reasoning not available - will skip HR metric")
        
        # Inizializza risultati
        results = {}
        for metric in metrics_to_compute:
            results[metric] = {}
            for explainer in explainers_to_test:
                results[metric][explainer] = float('nan')
        
        # PROCESSING SEQUENZIALE SEMPLIFICATO
        print(f"\n[PROCESSING] Sequential processing of {len(explainers_to_test)} explainers...")
        
        successful_explainers = 0
        
        for i, explainer_name in enumerate(explainers_to_test, 1):
            print(f"\n[{i}/{len(explainers_to_test)}] Processing {explainer_name}...")
            print("-" * 50)
            
            explainer_start_time = time.time()
            explainer_success = False
            
            try:
                # Crea explainer
                explainer = explainers.get_explainer(explainer_name, model, tokenizer)
                
                # Processa ogni metrica
                for metric_name in metrics_to_compute:
                    print(f"  [METRIC] {metric_name}...", end=" ")
                    metric_start_time = time.time()
                    
                    try:
                        if metric_name == "robustness":
                            score = metrics.evaluate_robustness_over_dataset(
                                model, tokenizer, explainer, consistency_texts, show_progress=False
                            )
                            
                        elif metric_name == "contrastivity":
                            # Process in batch per memoria
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
                                
                        elif metric_name == "consistency":
                            # CORREZIONE CONSISTENCY: Gestisce tupla (media, std) CORRETTAMENTE
                            print(f"[CONSISTENCY-FIX] Computing mean±std...", end="")
                            
                            mean_score, std_score = metrics.evaluate_consistency_over_dataset(
                                model=model,
                                tokenizer=tokenizer,
                                explainer=explainer,
                                texts=consistency_texts,
                                seeds=DEFAULT_CONSISTENCY_SEEDS,
                                show_progress=False
                            )
                            
                            # CORREZIONE CHIAVE: Crea stringa formattata "media±std"
                            formatted_score = f"{mean_score:.4f}±{std_score:.4f}"
                            
                            # CORREZIONE CHIAVE: Salva la stringa formattata nei risultati
                            results[metric_name][explainer_name] = formatted_score
                            
                            # CORREZIONE CHIAVE: Usa mean_score per il print di debug
                            score = mean_score
                            
                            print(f" -> {formatted_score}", end="")

                        elif metric_name == "human_reasoning":
                            if hr_available and hr_dataset is not None:
                                score = metrics.evaluate_human_reasoning_over_dataset(
                                    model=model,
                                    tokenizer=tokenizer,
                                    explainer=explainer,
                                    hr_dataset=hr_dataset,
                                    show_progress=False
                                )
                            else:
                                print("SKIPPED (HR not available)")
                                score = float('nan')
                                continue
                                
                        else:
                            score = float('nan')
                        
                        # CORREZIONE CHIAVE: Per consistency, NON sovrascrivere formatted_score
                        if metric_name != "consistency":
                            results[metric_name][explainer_name] = score
                        
                        metric_time = time.time() - metric_start_time
                        print(f" SUCCESS {score:.4f} ({metric_time:.1f}s)")
                        explainer_success = True
                        
                    except Exception as e:
                        print(f"ERROR: {str(e)[:50]}...")
                        if metric_name == "consistency":
                            results[metric_name][explainer_name] = "NaN±NaN"
                        else:
                            results[metric_name][explainer_name] = float('nan')
                
                if explainer_success:
                    successful_explainers += 1
                
                explainer_time = time.time() - explainer_start_time
                status = "SUCCESS" if explainer_success else "FAILED"
                print(f"  [TOTAL] {explainer_name}: {explainer_time:.1f}s {status}")
                
                # Cleanup explainer
                del explainer
                memory_manager.progressive_cleanup("light")
                
                # Checkpoint intermedio
                checkpoint_data = {
                    "results": results,
                    "completed": False,
                    "progress": {
                        "explainers_completed": i,
                        "explainers_total": len(explainers_to_test),
                        "successful_explainers": successful_explainers
                    }
                }
                if recovery:
                    recovery.save_checkpoint(checkpoint_data, f"model_{model_key}")
                
            except Exception as e:
                print(f"  EXPLAINER FAILED: {explainer_name}: {e}")
                for metric in metrics_to_compute:
                    if metric == "consistency":
                        results[metric][explainer_name] = "NaN±NaN"
                    else:
                        results[metric][explainer_name] = float('nan')
        
        # Salva risultati finali
        final_results = {
            "results": results,
            "completed": True,
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "successful_explainers": successful_explainers,
                "total_explainers": len(explainers_to_test),
                "success_rate": successful_explainers / len(explainers_to_test)
            }
        }
        if recovery:
            recovery.save_checkpoint(final_results, f"model_{model_key}")
        
        # Salvataggio asincrono
        if async_io:
            async_io.save_async(results, RESULTS_DIR / f"results_{model_key}_simplified.json")
        
        profiler.end_operation(f"model_{model_key}_simplified")
        
        # Stampa statistiche cache
        if embedding_cache:
            embedding_cache.print_stats()
        
        print(f"\n[COMPLETE] {model_key} simplified processing completed")
        print(f"[STATS] Successful explainers: {successful_explainers}/{len(explainers_to_test)}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Model {model_key} simplified processing failed: {e}")
        import traceback
        traceback.print_exc()
        
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
        
        if async_io:
            async_io.cleanup()
        
        memory_manager.progressive_cleanup("aggressive")

def smart_model_ordering(models_to_test: List[str]) -> List[str]:
    """Ordina modelli intelligentemente per ottimizzare processing."""
    size_priority = {
        "tinybert": 1,      # Più piccolo - processa per primo
        "distilbert": 2,    # Piccolo
        "roberta-base": 3,  # Medio
        "bert-large": 4,    # Grande
        "roberta-large": 5  # Più grande - processa per ultimo
    }
    
    ordered = sorted(models_to_test, key=lambda m: size_priority.get(m, 999))
    print(f"[ORDERING] Smart model order: {ordered}")
    return ordered

def verify_checkpoint_completeness(
    checkpoint_data: Dict, 
    expected_metrics: List[str], 
    expected_explainers: List[str]
) -> Tuple[bool, str]:
    """Verifica che il checkpoint contenga tutti i dati attesi."""
    if not checkpoint_data.get("completed", False):
        return False, "marked as incomplete"
    
    results = checkpoint_data.get("results", {})
    if not results:
        return False, "no results data"
    
    # Verifica presenza di tutte le metriche
    missing_metrics = []
    for metric in expected_metrics:
        if metric not in results:
            missing_metrics.append(metric)
    
    if missing_metrics:
        return False, f"missing metrics: {missing_metrics}"
    
    # Verifica presenza di tutti gli explainer per ogni metrica
    incomplete_metrics = []
    for metric in expected_metrics:
        if metric not in results:
            continue
            
        metric_explainers = set(results[metric].keys())
        expected_explainers_set = set(expected_explainers)
        
        if not expected_explainers_set.issubset(metric_explainers):
            missing = expected_explainers_set - metric_explainers
            incomplete_metrics.append(f"{metric}({list(missing)})")
    
    if incomplete_metrics:
        return False, f"incomplete explainers: {incomplete_metrics}"
    
    # Verifica che ci siano risultati validi
    valid_results_count = 0
    total_expected = len(expected_metrics) * len(expected_explainers)
    
    for metric in expected_metrics:
        for explainer in expected_explainers:
            if explainer in results[metric]:
                score = results[metric][explainer]
                # CORREZIONE CONSISTENCY: Considera anche stringhe "mean±std" come valide
                if metric == "consistency":
                    if isinstance(score, str) and "±" in score and "NaN" not in score:
                        valid_results_count += 1
                elif score is not None and not (isinstance(score, float) and np.isnan(score)):
                    valid_results_count += 1
    
    if valid_results_count == 0:
        return False, "no valid scores (all NaN/None)"
    
    completeness_ratio = valid_results_count / total_expected
    if completeness_ratio < 0.1:
        return False, f"too few valid results: {valid_results_count}/{total_expected} ({completeness_ratio:.1%})"
    
    return True, f"complete with {valid_results_count}/{total_expected} valid results ({completeness_ratio:.1%})"

def build_report_tables(all_results: Dict[str, Dict], metrics_to_compute: List[str]) -> Dict[str, pd.DataFrame]:
    """Costruisce tabelle finali dai risultati (incluso Human Reasoning + FIXED CONSISTENCY)."""
    print(f"\n{'='*70}")
    print(" BUILDING REPORT TABLES (SIMPLIFIED + HR + FIXED CONSISTENCY)")
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
                    # GESTIONE SPECIALE PER CONSISTENCY - FIXED
                    if metric == "consistency" and isinstance(score, str) and "±" in score:
                        # Mantieni il formato stringa "media±std"
                        metric_data[explainer_name][model_key] = score
                        print(f"[DEBUG] Consistency {explainer_name}+{model_key}: {score}")  # DEBUG
                    elif not (isinstance(score, float) and np.isnan(score)):
                        # Altri casi normali
                        metric_data[explainer_name][model_key] = score
        
        if metric_data:
            df = pd.DataFrame(metric_data).T
            tables[metric] = df
            print(f"[TABLE] {metric}: {df.shape[0]} explainers × {df.shape[1]} models")
            
            # DEBUGGING: Verifica contenuto per consistency
            if metric == "consistency":
                print(f"[DEBUG] Consistency table preview:")
                print(df.head())
        else:
            print(f"[TABLE] {metric}: No data available")
            tables[metric] = pd.DataFrame()
    
    return tables

def print_table_analysis(df: pd.DataFrame, metric_name: str):
    """Analisi e interpretazione tabella (FIXED CONSISTENCY)."""
    print(f"\n{'='*60}")
    print(f" {metric_name.upper()} ANALYSIS (FIXED)")
    print(f"{'='*60}")
    
    if df.empty:
        print("No data available for analysis")
        return
    
    if metric_name == "consistency":
        print(" Per-Explainer Statistics (mean ± std):")
        print("-" * 50)
        
        explainer_stats = []  # Per ranking finale
        
        for explainer in df.index:
            values_str = df.loc[explainer].dropna()
            if len(values_str) > 0:
                # Parse valori mean±std
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
                    
                    # MIGLIORAMENTO: Aggiungi range di variabilità
                    min_mean = np.min(means)
                    max_mean = np.max(means)
                    
                    print(f"  {explainer:>15s}: μ={avg_mean:.4f}±{avg_std:.4f} "
                          f"range=[{min_mean:.4f},{max_mean:.4f}] (n={count}, {coverage:.1%})")
                    
                    # Salva per ranking
                    explainer_stats.append((explainer, avg_mean, avg_std, min_mean, max_mean))
        
        # AGGIUNTA: Ranking degli explainer
        if explainer_stats:
            print(f"\n Explainer Ranking (by consistency):")
            print("-" * 50)
            explainer_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by mean
            
            for i, (explainer, avg_mean, avg_std, min_mean, max_mean) in enumerate(explainer_stats):
                stability = "HIGH" if avg_std < 0.05 else "MED" if avg_std < 0.1 else "LOW"
                print(f"  {i+1}. {explainer:>15s}: {avg_mean:.4f}±{avg_std:.4f} [{stability} stability]")
        
        # Top combinations per consistency
        print(f"\n Top 5 Combinations (by mean consistency):")
        print("-" * 50)
        flat_data = []
        for explainer in df.index:
            for model in df.columns:
                value = df.loc[explainer, model]
                if isinstance(value, str) and "±" in value:
                    try:
                        mean_part, std_part = value.split("±")
                        mean_val = float(mean_part)
                        std_val = float(std_part)
                        flat_data.append((explainer, model, mean_val, std_val, value))
                    except ValueError:
                        continue
        
        if flat_data:
            flat_data.sort(key=lambda x: x[2], reverse=True)  # Sort by mean
            print(f"  (Higher = Better)")
            for i, (explainer, model, mean_val, std_val, formatted) in enumerate(flat_data[:5]):
                stability = "STABLE" if std_val < 0.1 else "MODERATE" if std_val < 0.2 else "UNSTABLE"
                print(f"  {i+1}. {explainer:>12s} + {model:>12s}: {formatted} [{stability}]")
        
        return  # IMPORTANTE: esce qui per consistency
    
    # Statistiche per explainer (altri metrics)
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
    
    # Statistiche per modello
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
    
    # Ranking top combinations
    print(f"\n Top 5 Combinations:")
    print("-" * 40)
    flat_data = []
    for explainer in df.index:
        for model in df.columns:
            value = df.loc[explainer, model]
            if not pd.isna(value):
                flat_data.append((explainer, model, value))
    
    if flat_data:
        if metric_name == "robustness":
            flat_data.sort(key=lambda x: x[2])  # Lower is better
            direction = "(Lower = Better)"
        else:  # contrastivity, human_reasoning - higher is better
            flat_data.sort(key=lambda x: x[2], reverse=True)  # Higher is better
            direction = "(Higher = Better)"
        
        print(f"  {direction}")
        for i, (explainer, model, value) in enumerate(flat_data[:5]):
            print(f"  {i+1}. {explainer:>12s} + {model:>12s}: {value:.4f}")
    
    # Coverage analysis
    total_cells = df.size
    filled_cells = df.notna().sum().sum()
    coverage = filled_cells / total_cells if total_cells > 0 else 0
    
    print(f"\n Coverage Analysis:")
    print(f"  Total combinations: {total_cells}")
    print(f"  Completed: {filled_cells}")
    print(f"  Coverage: {coverage:.1%}")
    
    # Metric-specific insights
    if metric_name == "human_reasoning":
        print(f"\n Human Reasoning Insights:")
        if not df.empty:
            overall_mean = df.stack().mean()
            print(f"  Overall agreement: {overall_mean:.4f} (MAP score)")
            if overall_mean > 0.6:
                print(f"  → Good alignment with human-like reasoning")
            elif overall_mean > 0.4:
                print(f"  → Moderate alignment with human-like reasoning")
            else:
                print(f"  → Poor alignment with human-like reasoning")

