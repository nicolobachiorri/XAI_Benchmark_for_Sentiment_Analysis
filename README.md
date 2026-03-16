# XAI Benchmark for Sentiment Analysis

> Bachelor's thesis project — Università degli Studi di Milano-Bicocca  
> BSc in Statistics and Information Management

---

## Overview

This repository contains the code developed for my bachelor's thesis, which proposes a **systematic benchmark for evaluating post-hoc explainability methods** applied to encoder-only Transformer architectures in the context of **sentiment analysis**.

The benchmark assesses six XAI methods across three quantitative metrics — robustness, consistency, and contrastivity — providing a structured comparison of their reliability and discriminative power on pre-trained language models.

---

## XAI Methods Evaluated

| Method | Category |
|--------|----------|
| LIME | Perturbation-based |
| SHAP | Perturbation-based |
| Input × Gradient  | Gradient-based |
| Layer-wise Relevance Propagation (LRP) | Gradient-based |
| Attention Rollout | Attention-based |
| Attention Flow | Attention-based |

---

## Evaluation Metrics

### Robustness
Measures explanation stability under local text perturbations (token masking, deletion, substitution). Lower divergence indicates higher robustness.

### Consistency
Measures explanation stability across inference seeds. For each observation, pairwise correlations between explanations generated under different seeds are computed; the final score is the mean correlation across the dataset.

### Contrastivity
Measures the capacity of an explainer to produce distinguishable attributions for positive vs. negative sentiment instances, via KL divergence over token-level attribution distributions.

---

## Models

Five pre-trained Transformer models sourced from HuggingFace:

| Key | HuggingFace Model |
|-----|-------------------|
| `tinybert` | [Harsha901/tinybert-imdb-sentiment-analysis-model](https://huggingface.co/Harsha901/tinybert-imdb-sentiment-analysis-model) |
| `distilbert` | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) |
| `roberta-base` | [AnkitAI/reviews-roberta-base-sentiment-analysis](https://huggingface.co/AnkitAI/reviews-roberta-base-sentiment-analysis) |
| `roberta-large` | [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english) |
| `bert-large` | [assemblyai/bert-large-uncased-sst2](https://huggingface.co/assemblyai/bert-large-uncased-sst2) |

---

## Dataset

The benchmark uses the [IMDB Dataset for Sentiment Analysis](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format) (CSV format) sourced from Kaggle, containing movie reviews labeled as positive or negative. Sampling is performed via K-Means clustering to ensure lexical diversity and class balance across the selected observations (default: 400 samples).

---

## Project Structure

```
XAI-benchmark-Thesis/
├── models.py           # Model loading and GPU management
├── dataset.py          # Clustering-based stratified sampling
├── explainers.py       # XAI method implementations
├── metrics.py          # Robustness, consistency, contrastivity
├── report.py           # Main benchmark runner
├── utils.py            # Memory management utilities
├── Train.csv           # Training split
├── Test.csv            # Test split
└── requirements.txt    # Dependencies
```

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

> Designed for **Google Colab** with GPU support (12–16 GB VRAM recommended).

### Run the Benchmark

```bash
# Full benchmark with default settings (400 samples, all models and explainers)
python report.py --sample 400

# Custom model and explainer selection
python report.py --sample 400 --models tinybert distilbert --explainers lime shap grad_input

# Specific metrics only
python report.py --sample 200 --metrics robustness consistency
```

Results are saved as CSV tables in the `xai_results/` directory.

---

## Dependencies

```
torch>=2.2.0
transformers>=4.41.0
pandas>=2.2.2
scikit-learn>=1.5.0
lime>=0.2.0
shap>=0.45.0
networkx>=3.0
```

See `requirements.txt` for the full list.

---

## Author

**Nicolò Bachiorri**  
BSc in Statistics and Information Management  
Università degli Studi di Milano-Bicocca

