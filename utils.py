"""
utils.py – Essential utilities for Google Colab XAI Benchmark
===========================================================

"""

import json
import random
import time
import gc
import os
import warnings
from pathlib import Path
from typing import Any, List, TypeVar, Optional, Dict
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm

T = TypeVar("T")

# ==== Environment Setup ====
def setup_colab_environment():
    """Setup ottimale per Google Colab."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    try:
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        pass

# Auto setup
setup_colab_environment()

# ==== Seed Management ====
def set_seed(seed: int) -> None:
    """Fissa seed per riproducibilità completa."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==== Memory Management ====
def get_memory_usage() -> Dict[str, float]:
    """Ottieni uso memoria corrente."""
    usage = {}
    
    # RAM
    try:
        import psutil
        ram_info = psutil.virtual_memory()
        usage["ram_used_gb"] = ram_info.used / (1024**3)
        usage["ram_total_gb"] = ram_info.total / (1024**3)
        usage["ram_percent"] = ram_info.percent
    except ImportError:
        usage["ram_used_gb"] = 0.0
        usage["ram_total_gb"] = 0.0
        usage["ram_percent"] = 0.0
    
    # GPU
    if torch.cuda.is_available():
        usage["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        usage["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        usage["gpu_percent"] = (usage["gpu_allocated_gb"] / usage["gpu_total_gb"]) * 100
    else:
        usage["gpu_allocated_gb"] = 0.0
        usage["gpu_total_gb"] = 0.0
        usage["gpu_percent"] = 0.0
    
    return usage

def print_memory_status():
    """Stampa status memoria."""
    usage = get_memory_usage()
    print(f"Memory - RAM: {usage['ram_used_gb']:.1f}GB ({usage['ram_percent']:.1f}%) "
          f"GPU: {usage['gpu_allocated_gb']:.1f}GB ({usage['gpu_percent']:.1f}%)")

def aggressive_cleanup():
    """Cleanup aggressivo per Colab."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# ==== Performance Tracking ====
class Timer:
    """Timer semplice con memoria."""
    
    def __init__(self, name: str = "Operation", track_memory: bool = True):
        self.name = name
        self.track_memory = track_memory
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        if self.track_memory:
            self.start_memory = get_memory_usage()
        print(f"[START] {self.name}...")
        return self
        
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        print(f"[DONE] {self.name} completed in {self.format_time(duration)}")
        
        if self.track_memory and self.start_memory:
            end_memory = get_memory_usage()
            ram_delta = end_memory["ram_used_gb"] - self.start_memory["ram_used_gb"]
            gpu_delta = end_memory["gpu_allocated_gb"] - self.start_memory["gpu_allocated_gb"]
            
            if abs(ram_delta) > 0.1 or abs(gpu_delta) > 0.1:
                print(f"[MEMORY] RAM: {ram_delta:+.2f}GB, GPU: {gpu_delta:+.2f}GB")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Formatta tempo."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            return f"{h:.0f}h {m:.0f}m"

class PerformanceProfiler:
    """Profiler per tracking performance."""
    
    def __init__(self):
        self.timings = {}
    
    def start_operation(self, name: str):
        """Inizia tracking operazione."""
        self.timings[name] = {
            "start": time.time(),
            "memory_start": get_memory_usage()
        }
    
    def end_operation(self, name: str):
        """Termina tracking operazione."""
        if name in self.timings:
            self.timings[name]["end"] = time.time()
            self.timings[name]["memory_end"] = get_memory_usage()
            self.timings[name]["duration"] = self.timings[name]["end"] - self.timings[name]["start"]
    
    def print_summary(self):
        """Stampa summary performance."""
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        for name, data in self.timings.items():
            if "duration" in data:
                ram_delta = data["memory_end"]["ram_used_gb"] - data["memory_start"]["ram_used_gb"]
                gpu_delta = data["memory_end"]["gpu_allocated_gb"] - data["memory_start"]["gpu_allocated_gb"]
                
                print(f"{name:30s} {Timer.format_time(data['duration']):>10s} "
                      f"RAM: {ram_delta:+6.2f}GB GPU: {gpu_delta:+6.2f}GB")
        
        print(f"{'='*60}")

# ==== Essential File Operations ====
def save_json(path: str, obj: Any) -> None:
    """Salva JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path: str) -> Any:
    """Carica JSON."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# ==== Initialize ====
if __name__ == "__main__":
    print("Testing essential utilities...")
    set_seed(42)
    print_memory_status()
    
    with Timer("Test operation"):
        time.sleep(0.1)
    
    print("✓ Essential utilities working")
else:
    set_seed(42)