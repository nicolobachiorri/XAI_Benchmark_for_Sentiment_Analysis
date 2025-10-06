# XAI Benchmark for Sentiment Analysis

A  benchmark for evaluating explainable AI methods on sentiment analysis models. Designed for Google Colab with GPU support.

## Overview

This benchmark evaluates XAI methods across three core metrics:

- **Robustness**: Stability under text perturbations
- **Consistency**: Stability across inference seeds with correct per-observation correlation logic
- **Contrastivity**: Distinguishability between positive and negative sentiment classes

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run complete benchmark with default settings (400 samples)
python report.py --sample 400

# Custom configuration
python report.py --sample 400 --models tinybert distilbert --explainers lime shap grad_input

# Specific metrics only
python report.py --sample 200 --metrics robustness consistency
```

## Core Components

### Models (`models.py`)
Pre-trained sentiment analysis models with automatic GPU setup:
- `tinybert`: Lightweight BERT variant
- `distilbert`: DistilBERT for SST-2
- `roberta-base`: RoBERTa base model
- `roberta-large`: RoBERTa large model
- `bert-large`: BERT large for SST-2

### Dataset (`dataset.py`)
Optimized clustering-based sampling:
- Automatic K-means clustering for diversity
- Stratified sampling ensuring class balance
- Target size: 400 observations (configurable)
- Memory-efficient processing

### Explainers (`explainers.py`)
Six XAI methods with robust error handling:
- `lime`: LIME text explanations
- `shap`: SHAP kernel explainer
- `grad_input`: Gradient × Input attribution
- `attention_rollout`: Attention rollout method
- `attention_flow`: Attention flow analysis
- `lrp`: Layer-wise Relevance Propagation

### Metrics (`metrics.py`)
Three evaluation metrics:

#### Robustness
Measures stability under perturbations:
- Random masking, deletion, substitution
- Lower scores indicate higher robustness

#### Consistency 
Measures stability across inference seeds:
1. For each observation: generate explanations with multiple seeds
2. Compute all pairwise correlations between seed explanations
3. Calculate mean correlation per observation
4. Final metric: mean ± std across all observations

#### Contrastivity
Measures distinguishability between sentiment classes:
- KL divergence
- Token-level attribution distributions
- Higher scores indicate better contrastivity

## Configuration

### Memory Management
Automatic optimization for Google Colab:
- Dynamic batch sizing based on available memory
- Aggressive cleanup between operations
- GPU memory monitoring

### Reproducibility
- Fixed random seeds (42)
- Deterministic CUDA operations
- Consistent tokenization

## Output

The benchmark generates:
- CSV tables for each metric
- Results saved in `xai_results/` directory


## Dependencies

Core requirements:
- `torch>=2.2.0`
- `transformers>=4.41.0`
- `pandas>=2.2.2`
- `scikit-learn>=1.5.0`
- `lime>=0.2.0`
- `shap>=0.45.0`
- `networkx>=3.0`

See `requirements.txt` for complete list.

## Architecture

```
├── models.py          # Model loading and GPU management
├── dataset.py         # Clustering-based sampling
├── explainers.py      # XAI method implementations
├── metrics.py         # Evaluation metrics
├── report.py          # Main benchmark runner
├── utils.py           # Utilities and memory management
└── requirements.txt   # Dependencies
```



## Performance Considerations

- **Memory**: Optimized for 12-16GB GPU memory
- **Speed**: ~0.8 minutes per observation for consistency
- **Scaling**: Automatic batch size adjustment
- **Cleanup**: Aggressive memory management between operations




