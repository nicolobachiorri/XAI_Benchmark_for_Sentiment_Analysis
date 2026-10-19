# XAI Benchmark for Sentiment Analysis

Code for my bachelor's thesis *Valutazione di metodi XAI applicati a modelli linguistici basati su encoder per l'analisi del sentiment* (Università degli Studi di Milano-Bicocca, BSc in Statistics and Information Management, A.Y. 2024/2025). Supervisor: Prof. Navid Nobani. Co-supervisor: Dr. Antonio Serino. The thesis is in [`triennale_895103.pdf`](triennale_895103.pdf).

The benchmark compares six post-hoc explanation methods on five fine-tuned encoder-only Transformers, using three metrics: robustness, consistency and contrastivity.

## Two versions of the code

| | `thesis-v1` (tag) | `v2` (this branch) |
|---|---|---|
| Commit | `fc73a9e`, 22 Aug 2025 | current |
| Produced the thesis tables | yes (run of 22 Aug 2025, saved in [`results_thesis_v1/`](results_thesis_v1)) | no: own run of 15 Oct 2026, saved in [`results_v2/`](results_v2) |
| Matches the methods described in the thesis | partly, see [Known issues in v1](#known-issues-in-v1) | yes, with the choices listed in [Choices not fixed by the thesis](#choices-not-fixed-by-the-thesis) |

The numbers in the thesis come from `thesis-v1`. A later clean-up (September 2025) removed unused code and broke `report.py`, but did not change the explainers or the logic of the metrics. `v2` rewrites them to follow the definitions in chapters 2 and 3 of the thesis. Because the implementations change, the `v2` numbers differ from the thesis tables; see [Results (v2)](#results-v2).

## Methods

| Method | Family | Thesis | v2 implementation |
|---|---|---|---|
| LIME | perturbation | pp. 31-33 | `lime` library, one feature per word position, removed words replaced by the mask token |
| KernelSHAP | perturbation | pp. 33-34, eq. 2.7-2.10 | own implementation of eq. 2.9-2.10 (`attribution.kernel_shap_values`), 15 sampled coalitions, mask-token baseline |
| Input×Gradient | gradient | p. 35, eq. 2.11 | gradient of the positive-class logit times word embeddings |
| LRP | relevance propagation | pp. 36-38, eq. 2.18-2.19 | Ali et al. (2022) AH-rule and LN-rule: attention weights and the LayerNorm scale treated as constants, then Gradient×Input on the modified graph |
| Attention Rollout | attention | pp. 38-39, eq. 2.20-2.24 | `A = ½W + ½I`, product over layers, `[CLS]` row |
| Attention Flow | attention | pp. 39-40 | exact max-flow (Dinic) from `[CLS]` in the last layer to every input token |

All explanations are reported per word (scores of sub-word tokens are summed), so the six methods are compared on the same unit.

## Metrics

- **Robustness** (eq. 3.1-3.3): mean absolute difference of [0, 1]-normalised scores on the words shared by the explanation of a text and of a perturbed copy (random masking, deletion, synonym substitution, ρ = 0.15, two perturbations each). Lower is more stable.
- **Consistency** (eq. 3.4-3.6): dropout switched on at inference, four seeds (42, 123, 456, 789), mean Spearman correlation over the six seed pairs, then mean ± standard deviation over texts.
- **Contrastivity** (p. 50): KL divergence, base 2, between the per-word score distributions of positive and negative texts, over the 1,000 most frequent words.

Robustness and consistency measure stability, not faithfulness: a method that always returned the same explanation would score perfectly on both.

## Models and data

| Key | Hugging Face model | Layers |
|---|---|---|
| `tinybert` | [Harsha901/tinybert-imdb-sentiment-analysis-model](https://huggingface.co/Harsha901/tinybert-imdb-sentiment-analysis-model) | 4 |
| `distilbert` | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | 6 |
| `roberta-base` | [AnkitAI/reviews-roberta-base-sentiment-analysis](https://huggingface.co/AnkitAI/reviews-roberta-base-sentiment-analysis) | 12 |
| `bert-large` | [assemblyai/bert-large-uncased-sst2](https://huggingface.co/assemblyai/bert-large-uncased-sst2) | 24 |
| `roberta-large` | [siebert/sentiment-roberta-large-english](https://huggingface.co/siebert/sentiment-roberta-large-english) | 24 |

The models were fine-tuned by their authors on different data (SST-2, IMDB, other review corpora), so model size and training data are confounded.

Data: 400 reviews drawn from the 5,000 reviews in `Test.csv` (IMDB) with TF-IDF + k-means and stratified sampling (thesis, p. 50): 188 positive, 212 negative. The sample is frozen in [`data/imdb_clustered_400.csv`](data/imdb_clustered_400.csv), because k-means results depend on the scikit-learn version. Rebuilding it with `python dataset.py --rebuild` gives the same class counts as in the thesis. `Train.csv` is not used.

## Running

Requirements: Python 3.10+, PyTorch, Transformers 4.46 or later (the code needs `attn_implementation="eager"` and fast tokenizers), `lime`, NumPy, SciPy, scikit-learn, pandas. See [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt

# 1. tests (CPU, a few minutes, no download): tiny random BERT/RoBERTa/DistilBERT
python -m pytest -q tests

# 2. time estimate on the current hardware, before paying for a long run
python report.py --estimate

# 3. full benchmark (400 texts, 5 models, 6 methods, 3 metrics)
python report.py --out-dir runs/v2
```

The run can be interrupted and restarted with the same command: each finished text is written immediately and skipped on restart.

To use all the hardware at once, run the benchmark through the parallel launcher:

```bash
python parallel_run.py --out-dir runs/v2            # add --dry-run to see the plan
```

It splits the texts into shards and runs several `report.py` processes together: two on the GPU (one if the machine has two CPU cores or fewer) for LIME, SHAP, Input×Gradient, Rollout and LRP, and one per free CPU core for Attention Flow, whose max-flow step runs on the CPU only (deeper models get more shards). Every text has its own seeds, so the results are the same as a single sequential run; this is checked by `tests/test_torch.py`. At the end the launcher rebuilds the tables with `report.py --summarize-only`.

On Google Colab, [`colab_run.ipynb`](colab_run.ipynb) contains these steps: tests on a free CPU runtime, `--estimate` on a GPU runtime, then the parallel run with `--out-dir` on Google Drive so that results survive a disconnection. If the GPU runtime has few CPU cores, the work can be split between two runtimes on the same Drive folder: `parallel_run.py --role gpu` on the GPU runtime and `parallel_run.py --role flow` on a CPU runtime, then `report.py --summarize-only`.

Main options of `report.py`: `--models`, `--explainers`, `--metrics`, `--sample` (default 400), `--max-length` (512), `--flow-max-length` (128), `--lime-samples` (500), `--shap-samples` (15), `--batch-size`, `--device`, `--shard I/N`, `--summarize-only`, `--no-resume`. Options that `parallel_run.py` does not know are passed to `report.py`.

Outputs in `--out-dir`: `<model>/<method>/eval*.jsonl` and `train*.jsonl` (per-text values, one file per shard), `summary.json` (aggregates and counts of invalid values), `robustness_table.csv`, `consistency_table.csv`, `contrastivity_table.csv`, `contrastivity_js_table.csv`, `run_metadata/` (configuration, library versions and git commit of each process), `logs/` (one log per process of the launcher).

## Choices not fixed by the thesis

- **Unit of explanation**: words separated by spaces; sub-word scores are summed.
- **Explained class**: always the positive class (positive logit for gradient methods, positive probability for LIME and SHAP).
- **Truncation**: 512 tokens for all methods except Attention Flow, which uses the first 128 tokens. Exact max-flow costs roughly L·n³: on one CPU core it takes about 3 s per explanation for a 24-layer model at 128 tokens and an estimated 4 minutes at 512, which would make the full run impractical. 46 of the 400 reviews have more than 380 words, about the 512-token limit, so for those the other methods also see only the beginning.
- **LIME samples**: 500 (the thesis does not state a number; the library default is 5,000).
- **Robustness**: two perturbations per strategy; a perturbation that cannot be applied (for example no word in the synonym dictionary) is recorded and excluded, not scored. The synonym dictionary has 130 entries.
- **Contrastivity**: scores are divided by the maximum absolute score of each explanation before summing. This keeps the sign, which the "S − min S" step needs, and gives each text the same weight. With [0, 1] scores, every word absent from one class would get probability zero and the divergence would be dominated by ε. The comparison uses the true labels of the texts. By construction of the formula, words with the minimum score in the negative class get probability zero, so part of the KL value comes from ε alone (reported as `kl_from_zero_neg` in `summary.json`). This is strongest for Rollout and Flow, whose scores are never negative. For this reason the Jensen-Shannon divergence, which the thesis mentions as the symmetric alternative (p. 50), is also reported (`contrastivity_js_table.csv`): it is bounded in [0, 1] and needs no ε.
- **Invalid values** (for example Spearman on a constant explanation): `NaN`, excluded from the means and counted in `summary.json`.
- **Precision**: weights in float32; float16 autocast only for the gradient-free forward passes of LIME and SHAP on GPU.

## Known issues in v1

Found when comparing `thesis-v1` with the thesis text. They affect the tables in the thesis.

- `report.py` (after the September clean-up) fails at start-up: `main()` passes `backup_to_drive` to a function without that parameter.
- Robustness and consistency used at most 40 texts, contrastivity 40 per class (fewer with less free RAM), not 400.
- LIME used 50 samples and only the 15 most important words; these settings are not stated in the thesis.
- LRP was a single backward pass with `e·g/(|e|+ε)`, without the AH and LN rules: in practice Input×Gradient, which is why the two methods have almost identical numbers.
- Attention Flow computed the max-flow for the first 5 positions only, on edges above 0.1, so most tokens got zero.
- SHAP: the KernelSHAP code with 15 samples was after a `return` and never ran; `shap.Explainer` with a text masker ran instead, truncated at 128 tokens.
- Contrastivity kept the first 1,000 tokens in alphabetical order, not the most frequent.
- Consistency switched dropout on, but four explainers called `model.eval()` at each explanation, so for them dropout was off; only LIME and SHAP were evaluated with dropout.
- Consistency matched tokens by string, keeping only the last occurrence of repeated tokens: two identical explanations of a text with repeated words did not get a correlation of 1.
- Robustness: the synonym dictionary had 12 entries and in most texts the substitution changed nothing; any error returned 0.0, the best possible score. Several fallbacks returned fixed values (0.05, 0.1) instead of missing values.
- Scores were normalised to [−1, 1] for four methods and left raw for LIME and SHAP, so the metrics compared different scales.
- Text, p. 52: SHAP on RoBERTa-large is reported as 0.0012, the table says 0.0117.

## Results (v2)

One run of `parallel_run.py` on a Colab A100 (12 CPU cores), 15 October 2026, 6.8 hours: 400 texts, 5 models, 6 methods, 3 metrics, no failed text. Per-text values are not in the repository (several hundred MB); the tables are in [`results_v2/`](results_v2).

Robustness (lower is more stable)

| Method | tinybert | distilbert | roberta-base | bert-large | roberta-large |
|---|---|---|---|---|---|
| LIME | 0.0951 | 0.1268 | 0.1404 | 0.1176 | 0.1537 |
| SHAP | 0.1663 | 0.1415 | 0.1500 | 0.1703 | 0.1044 |
| Input×Gradient | 0.1448 | 0.1326 | 0.1443 | 0.1609 | 0.2186 |
| Attention Rollout | 0.0365 | 0.0237 | 0.0263 | 0.0416 | 0.0185 |
| Attention Flow | 0.0202 | 0.0094 | 0.0104 | 0.0094 | 0.0120 |
| LRP | 0.0831 | 0.0948 | 0.0799 | 0.1643 | 0.0790 |

Consistency (mean ± sd over 400 texts)

| Method | tinybert | distilbert | roberta-base | bert-large | roberta-large |
|---|---|---|---|---|---|
| LIME | 0.2548 ± 0.1453 | 0.1824 ± 0.1296 | 0.1419 ± 0.1130 | 0.2045 ± 0.1359 | 0.1115 ± 0.1018 |
| SHAP | 0.0095 ± 0.0358 | 0.0093 ± 0.0356 | 0.0092 ± 0.0358 | 0.0055 ± 0.0378 | 0.0066 ± 0.0357 |
| Input×Gradient | 0.3642 ± 0.0988 | 0.4024 ± 0.0597 | 0.1893 ± 0.0679 | 0.2261 ± 0.0911 | 0.0999 ± 0.0766 |
| Attention Rollout | 0.8613 ± 0.0509 | 0.9133 ± 0.0256 | 0.9291 ± 0.0196 | 0.9362 ± 0.0152 | 0.8184 ± 0.0454 |
| Attention Flow | 0.9397 ± 0.0184 | 0.9201 ± 0.0224 | 0.9477 ± 0.0132 | 0.9464 ± 0.0114 | 0.9554 ± 0.0128 |
| LRP | 0.7884 ± 0.0475 | 0.8618 ± 0.0384 | 0.8135 ± 0.1225 | 0.3328 ± 0.1456 | 0.8497 ± 0.0747 |

Contrastivity, KL divergence as defined in the thesis (higher is more contrastive; affected by ε, see [Choices](#choices-not-fixed-by-the-thesis))

| Method | tinybert | distilbert | roberta-base | bert-large | roberta-large |
|---|---|---|---|---|---|
| LIME | 0.0350 | 0.0340 | 0.0351 | 0.0385 | 0.0283 |
| SHAP | 2.3351 | 1.2276 | 0.4125 | 1.5574 | 0.0581 |
| Input×Gradient | 0.0155 | 0.0220 | 0.0735 | 0.0254 | 0.0559 |
| Attention Rollout | 0.3346 | 0.3603 | 0.3201 | 0.2792 | 0.2811 |
| Attention Flow | 0.6469 | 0.6267 | 0.6605 | 0.6295 | 0.6253 |
| LRP | 0.0753 | 0.1446 | 1.2319 | 0.0003 | 1.7753 |

Contrastivity, Jensen-Shannon divergence (bounded in [0, 1], no ε)

| Method | tinybert | distilbert | roberta-base | bert-large | roberta-large |
|---|---|---|---|---|---|
| LIME | 0.0054 | 0.0043 | 0.0057 | 0.0059 | 0.0037 |
| SHAP | 0.1888 | 0.0876 | 0.0286 | 0.1906 | 0.0049 |
| Input×Gradient | 0.0023 | 0.0033 | 0.0054 | 0.0018 | 0.0097 |
| Attention Rollout | 0.0662 | 0.0539 | 0.0493 | 0.0450 | 0.0467 |
| Attention Flow | 0.0703 | 0.0654 | 0.0674 | 0.0655 | 0.0652 |
| LRP | 0.0154 | 0.0274 | 0.1523 | 0.0001 | 0.1926 |

### Reading the results

- **Attention Flow and Attention Rollout** are the most stable methods. Flow is first on robustness and consistency for every model; Rollout is second, except for consistency on roberta-large, where LRP is slightly higher (0.850 against 0.818). This agrees with the thesis.
- **LRP and Input×Gradient now differ.** With the AH and LN rules, LRP is more robust and much more consistent than Input×Gradient on four models out of five. In v1 the two methods were almost identical because LRP was in practice Input×Gradient.
- **SHAP with 15 coalitions is close to noise.** Its consistency is about 0.01 on every model: with 15 samples for a few hundred words, two runs with different seeds give unrelated explanations. The thesis setting (15 samples) is kept; a reliable SHAP estimate would need far more samples.
- **LIME (500 samples)** has low consistency (0.11 to 0.25), for the same reason on a smaller scale.
- **Contrastivity mostly reflects vocabulary.** If every word gets the same score, the metric only sees how often words occur in positive and negative reviews. This baseline gives a Jensen-Shannon divergence of 0.036 on the first 380 words and 0.053 on the first 100 (about the 128 tokens seen by Flow). Rollout (0.045 to 0.066) and Flow (0.065 to 0.070) are only slightly above it, because their scores are never negative and change little from word to word. Signed methods (Input×Gradient, LIME) fall below the baseline because positive and negative contributions cancel in the sums. The metric, as defined, does not separate the explanation from word frequency.
- **LRP on bert-large is an open anomaly.** Robustness 0.164, consistency 0.33 and contrastivity close to zero, while LRP works well on the other four models, including tinybert and distilbert from the same BERT family. Comparing the per-text explanations with roberta-large (also 24 layers): relevance is equally spread over the words in both models, but on bert-large its sum has a median of −2.5 on negative reviews and −0.35 on positive ones (roberta-large: −192 and +57), and the most relevant word is most often a contraction split into several tokens (`it's`, `don't`), not a sentiment word. One hypothesis, a saturated tanh in the BERT pooler blocking the gradient, was tested and rejected: on 15 reviews only 5% of the pooler units are saturated and the summed relevance is 0.42 times the logit. The cause is not known yet.

Compared with the thesis, the ranking on stability holds (attention methods first), while the conclusions on contrastivity do not: in v1 Input×Gradient and LRP looked the most contrastive and the attention methods the least, partly because of the alphabetical vocabulary and the 40-text cap.

## Results reported in the thesis (v1)

Produced by `thesis-v1` on 22 August 2025 (Colab). See the issues above before reading them.

Robustness (lower is more stable)

| Method | tinybert | distilbert | roberta-base | bert-large | roberta-large |
|---|---|---|---|---|---|
| LIME | 0.0461 | 0.0774 | 0.0581 | 0.0594 | 0.0621 |
| SHAP | 0.0049 | 0.0098 | 0.0082 | 0.0096 | 0.0117 |
| Input×Gradient | 0.2985 | 0.2809 | 0.3048 | 0.3552 | 0.4592 |
| Attention Rollout | 0.0117 | 0.0061 | 0.0034 | 0.0100 | 0.0048 |
| Attention Flow | 0.0957 | 0.0964 | 0.0750 | 0.0972 | 0.0751 |
| LRP | 0.2852 | 0.2906 | 0.3173 | 0.3639 | 0.4967 |

Consistency (mean ± sd)

| Method | tinybert | distilbert | roberta-base | bert-large | roberta-large |
|---|---|---|---|---|---|
| LIME | 0.2448 ± 0.3139 | 0.1781 ± 0.3058 | 0.0853 ± 0.3114 | 0.2992 ± 0.3348 | 0.0616 ± 0.3434 |
| SHAP | 0.4012 ± 0.1244 | 0.4543 ± 0.1425 | 0.3912 ± 0.1239 | 0.3423 ± 0.1548 | 0.2671 ± 0.1295 |
| Input×Gradient | 0.7602 ± 0.0712 | 0.6109 ± 0.1006 | 0.6977 ± 0.0903 | 0.6628 ± 0.1093 | 0.6652 ± 0.0956 |
| Attention Rollout | 0.8575 ± 0.0618 | 0.8536 ± 0.0623 | 0.8358 ± 0.0703 | 0.9550 ± 0.0285 | 0.8953 ± 0.0527 |
| Attention Flow | 0.7580 ± 0.1290 | 0.7540 ± 0.1284 | 0.7900 ± 0.1226 | 0.7538 ± 0.1246 | 0.7843 ± 0.1268 |
| LRP | 0.7833 ± 0.0566 | 0.6164 ± 0.1003 | 0.7047 ± 0.0855 | 0.6438 ± 0.1139 | 0.6657 ± 0.0955 |

Contrastivity (KL divergence, higher is more contrastive)

| Method | tinybert | distilbert | roberta-base | bert-large | roberta-large |
|---|---|---|---|---|---|
| LIME | 0.0434 | 0.0492 | 0.0886 | 0.0428 | 0.0414 |
| SHAP | 0.1636 | 0.0561 | 0.0641 | 0.0779 | 0.0400 |
| Input×Gradient | 0.8019 | 0.1464 | 0.0890 | 0.0123 | 0.4540 |
| Attention Rollout | 0.0002 | 0.0002 | 0.0040 | 0.0002 | 0.0038 |
| Attention Flow | 0.0002 | 0.0002 | 0.0014 | 0.0002 | 0.0014 |
| LRP | 0.7214 | 0.1170 | 0.0857 | 0.0032 | 0.2334 |

## Limitations (thesis, p. 61)

- 400 English reviews from IMDB only.
- The perturbations used for robustness do not cover all meaning-preserving transformations.
- Consistency is estimated within one model (dropout at inference) rather than across models fine-tuned with different seeds.
- Contrastivity is restricted to the 1,000 most frequent words, which lowers sensitivity to rare words.

## Repository structure

```
├── attribution.py      # word-level explanations, Rollout, Flow, KernelSHAP, LIME helper (no PyTorch)
├── explainers.py       # the six methods on Hugging Face models
├── metrics.py          # perturbations, robustness, consistency, contrastivity (no PyTorch)
├── models.py           # model registry and loading
├── dataset.py          # 400-review sample (k-means + stratified sampling)
├── report.py           # benchmark runner, time estimate, tables
├── parallel_run.py     # runs report.py in parallel on GPU and CPU cores
├── utils.py            # seeds, timing, result files
├── tests/              # test_core.py (NumPy only), test_torch.py (tiny random models)
├── colab_run.ipynb     # tests, time estimate and parallel run on Colab
├── data/imdb_clustered_400.csv
├── results_thesis_v1/  # tables and per-model results of the thesis run
├── results_v2/         # tables of the v2 run
├── Test.csv, Train.csv
└── triennale_895103.pdf
```
