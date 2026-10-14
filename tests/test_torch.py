"""Test con PyTorch su modelli finti (pesi casuali, pochi layer), senza scaricare nulla.

Si eseguono su CPU in pochi minuti, per esempio in un runtime Colab gratuito:

    pip install -q lime pytest
    python -m pytest -q tests

Verificano che il codice giri su BERT, RoBERTa e DistilBERT e che le regole della
tesi siano implementate davvero: LRP con AH-rule e LN-rule (conservazione della
rilevanza), dropout acceso solo nella consistency, seed riproducibili, pipeline
completa con ripresa dopo un'interruzione.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import dataset  # noqa: E402
import explainers  # noqa: E402
import metrics  # noqa: E402
import models  # noqa: E402
from attribution import normalize_whitespace  # noqa: E402
from utils import read_jsonl, set_seed  # noqa: E402

models.set_device("cpu")
MAXLEN = 64
CFG = explainers.ExplainerConfig(max_length=MAXLEN, flow_max_length=32, lime_samples=30,
                                 shap_samples=15, batch_size=16, amp=False)
_BUILT = {}


# =============================================================================
# Modelli e tokenizer finti
# =============================================================================
def _texts(n=200):
    return [normalize_whitespace(t) for t in dataset.get_clustered_sample(None)[0][:n]]


def _bert_tokenizer(texts, model_input_names=None):
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
    from tokenizers import models as tkm
    from transformers import PreTrainedTokenizerFast
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = Counter(w for t in texts for w in re.findall(r"\w+|[^\w\s]", t.lower()))
    chars = sorted({c for t in texts for c in t.lower() if not c.isspace()})
    vocab_list = specials + [w for w, _ in words.most_common(800)] + chars + ["##" + c for c in chars]
    vocab = {}
    for tok in vocab_list:
        vocab.setdefault(tok, len(vocab))
    tk = Tokenizer(tkm.WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    tk.decoder = decoders.WordPiece()
    extra = {"model_input_names": model_input_names} if model_input_names else {}
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", **extra)


def _roberta_tokenizer(texts):
    from tokenizers import Tokenizer, decoders, pre_tokenizers, processors, trainers
    from tokenizers import models as tkm
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(tkm.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=800, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(texts, trainer)
    tk.post_processor = processors.RobertaProcessing(sep=("</s>", tk.token_to_id("</s>")),
                                                     cls=("<s>", tk.token_to_id("<s>")),
                                                     trim_offsets=True, add_prefix_space=False)
    return PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                  pad_token="<pad>", cls_token="<s>", sep_token="</s>", mask_token="<mask>",
                                  model_input_names=["input_ids", "attention_mask"])


def _config(kind, tok, **over):
    from transformers import BertConfig, DistilBertConfig, RobertaConfig
    labels = {"id2label": {0: "NEGATIVE", 1: "POSITIVE"}, "label2id": {"NEGATIVE": 0, "POSITIVE": 1}}
    if kind == "bert":
        c = dict(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                 intermediate_size=64, max_position_embeddings=600, pad_token_id=tok.pad_token_id,
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        c.update(over)
        return BertConfig(**c, **labels)
    if kind == "roberta":
        c = dict(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                 intermediate_size=64, max_position_embeddings=600, pad_token_id=tok.pad_token_id,
                 bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, type_vocab_size=1,
                 hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        c.update(over)
        return RobertaConfig(**c, **labels)
    c = dict(vocab_size=len(tok), dim=32, n_layers=2, n_heads=2, hidden_dim=64, max_position_embeddings=600,
             pad_token_id=tok.pad_token_id, dropout=0.1, attention_dropout=0.1)
    c.update(over)
    return DistilBertConfig(**c, **labels)


def build_tiny_models(root: Path) -> dict:
    """Salva tre modelli finti (BERT, RoBERTa, DistilBERT) e li registra in models.MODELS."""
    if _BUILT:
        return _BUILT
    from transformers import AutoModelForSequenceClassification
    texts = _texts()
    torch.manual_seed(0)
    for key, kind in (("tiny-bert", "bert"), ("tiny-roberta", "roberta"), ("tiny-distilbert", "distilbert")):
        if kind == "roberta":
            tok = _roberta_tokenizer(texts)
        elif kind == "distilbert":
            tok = _bert_tokenizer(texts, model_input_names=["input_ids", "attention_mask"])
        else:
            tok = _bert_tokenizer(texts)
        model = AutoModelForSequenceClassification.from_config(_config(kind, tok), attn_implementation="eager")
        path = root / key
        model.save_pretrained(path)
        tok.save_pretrained(path)
        models.register_model(key, str(path))
        _BUILT[key] = str(path)
    return _BUILT


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return build_tiny_models(tmp_path_factory.mktemp("tiny_models"))


def _load(key):
    return models.load_model(key), models.load_tokenizer(key)


def _sample_text(tok, i=0):
    raw = dataset.get_clustered_sample(None)[0][i]
    return explainers.visible_text(normalize_whitespace(raw), tok, MAXLEN)


KEYS = ["tiny-bert", "tiny-roberta", "tiny-distilbert"]


# =============================================================================
# Caricamento
# =============================================================================
@pytest.mark.parametrize("key", KEYS)
def test_load_model_eager_float32_frozen(tiny, key):
    model, tok = _load(key)
    assert model.config._attn_implementation == "eager"
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters())
    assert not model.training
    assert models.positive_index(model) == 1
    assert tok.is_fast


# =============================================================================
# Explainer: forma dell'output e modalità del modello
# =============================================================================
@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("name", explainers.list_explainers())
def test_explainer_word_level_output_and_mode_untouched(tiny, key, name):
    model, tok = _load(key)
    text = _sample_text(tok)
    explain = explainers.get_explainer(name, model, tok, CFG)
    for mode in ("eval", "train"):
        getattr(model, mode)()
        set_seed(42)
        a = explain(text)
        assert model.training == (mode == "train"), f"{name} ha cambiato la modalità del modello"
        words = text.split(" ")
        assert a.words == words[:len(a.words)] and len(a.words) > 0
        if name != "attention_flow":
            assert len(a.words) == len(words)  # il testo è già troncato a MAXLEN
        s = np.asarray(a.scores)
        assert np.all(np.isfinite(s)) and s.min() >= 0 and s.max() <= 1
    model.eval()


def test_flow_uses_its_own_shorter_window(tiny):
    model, tok = _load("tiny-bert")
    text = _sample_text(tok)
    a = explainers.get_explainer("attention_flow", model, tok, CFG)(text)
    full = explainers.get_explainer("attention_rollout", model, tok, CFG)(text)
    assert len(a.words) < len(full.words)


# =============================================================================
# Input×Gradient
# =============================================================================
def test_grad_input_matches_manual_computation(tiny):
    model, tok = _load("tiny-bert")
    text = _sample_text(tok)
    a = explainers.get_explainer("grad_input", model, tok, CFG)(text)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAXLEN)
    emb = model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
    out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"], token_type_ids=enc["token_type_ids"])
    out.logits[0, 1].backward()
    per_token = (emb * emb.grad).sum(-1)[0]
    assert np.isclose(sum(a.raw_scores), float(per_token[1:-1].sum()), rtol=1e-4, atol=1e-6)


# =============================================================================
# LRP: AH-rule, LN-rule, conservazione
# =============================================================================
def _qk_v_names(model):
    if model.config.model_type == "distilbert":
        return ("q_lin.weight", "k_lin.weight"), ("v_lin.weight",)
    return ("query.weight", "key.weight"), ("value.weight",)


@pytest.mark.parametrize("key", KEYS)
def test_lrp_ah_rule_blocks_gradient_through_attention_weights(tiny, key):
    model, tok = _load(key)
    for p in model.parameters():
        p.requires_grad_(True)
    enc = tok(_sample_text(tok), return_tensors="pt", truncation=True, max_length=MAXLEN)
    enc = explainers._model_inputs(model, dict(enc))
    qk, v = _qk_v_names(model)

    def grads(use_rules):
        model.zero_grad(set_to_none=True)
        if use_rules:
            with explainers.lrp_rules():
                model(**enc).logits[0, 1].backward()
        else:
            model(**enc).logits[0, 1].backward()
        g_qk = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                   if n.endswith(qk) and p.grad is not None)
        g_v = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                  if n.endswith(v) and p.grad is not None)
        return g_qk, g_v

    qk_plain, _ = grads(False)
    qk_rules, v_rules = grads(True)
    assert qk_plain > 0
    assert qk_rules == 0.0, "con la AH-rule query e key non devono ricevere gradiente"
    assert v_rules > 0
    # le patch vengono rimosse all'uscita
    assert torch.nn.functional.softmax is explainers._ORIG_SOFTMAX
    assert torch.nn.functional.layer_norm is explainers._ORIG_LAYER_NORM


def test_ln_rule_same_forward_and_scale_treated_as_constant():
    torch.manual_seed(0)
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    w, b = torch.randn(8, dtype=torch.float64), torch.randn(8, dtype=torch.float64)
    y = explainers._layer_norm_const(x, (8,), w, b, 1e-5)
    assert torch.allclose(y, torch.nn.functional.layer_norm(x, (8,), w, b, 1e-5))
    c = torch.randn(3, 8, dtype=torch.float64)
    (y * c).sum().backward()
    std = torch.sqrt(x.detach().var(-1, unbiased=False, keepdim=True) + 1e-5)
    wc = w * c
    expected = (wc - wc.mean(-1, keepdim=True)) / std  # alpha costante, centratura derivata
    assert torch.allclose(x.grad, expected)


def _linearised_bert(tok):
    """BERT con ReLU, bias nulli e pooler senza tanh: con AH e LN rule la rete è
    omogenea di grado 1 nell'input, quindi la rilevanza deve conservarsi esattamente."""
    from transformers import AutoModelForSequenceClassification
    torch.manual_seed(1)
    cfg = _config("bert", tok, hidden_act="relu", hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                  initializer_range=0.5)
    m = AutoModelForSequenceClassification.from_config(cfg, attn_implementation="eager").double().eval()
    for n, p in m.named_parameters():
        p.requires_grad_(False)
        if n.endswith("bias"):
            p.zero_()
    m.bert.pooler.activation = torch.nn.Identity()
    return m


def _relevance_at_embedding_sum(model, enc, use_rules):
    captured = {}
    h = model.bert.embeddings.LayerNorm.register_forward_pre_hook(lambda mod, args: captured.update(x=args[0]))
    emb = model.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
    try:
        if use_rules:
            with explainers.lrp_rules():
                out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"], token_type_ids=enc["token_type_ids"])
                logit = out.logits[0, 1]
                (g,) = torch.autograd.grad(logit, emb)
        else:
            out = model(inputs_embeds=emb, attention_mask=enc["attention_mask"], token_type_ids=enc["token_type_ids"])
            logit = out.logits[0, 1]
            (g,) = torch.autograd.grad(logit, emb)
    finally:
        h.remove()
    # x = parola + posizione + tipo; il gradiente rispetto a x coincide con quello rispetto alla parola
    R = (captured["x"].detach() * g).sum()
    return float(R), float(logit)


def test_lrp_conserves_relevance_and_grad_input_does_not(tiny):
    tok = models.load_tokenizer("tiny-bert")
    m = _linearised_bert(tok)
    enc = tok(_sample_text(tok), return_tensors="pt", truncation=True, max_length=MAXLEN)
    R, f = _relevance_at_embedding_sum(m, enc, use_rules=True)
    assert abs(R - f) <= 1e-8 * max(1.0, abs(f)), (R, f)
    R0, f0 = _relevance_at_embedding_sum(m, enc, use_rules=False)
    assert abs(R0 - f0) > 1e-4 * max(1.0, abs(f0)), "senza le regole la conservazione non dovrebbe valere"


# =============================================================================
# Dropout e seed
# =============================================================================
@pytest.mark.parametrize("name", ["grad_input", "lrp", "attention_rollout", "lime", "shap"])
def test_dropout_only_in_train_mode_and_seeds_reproducible(tiny, name):
    model, tok = _load("tiny-bert")
    text = _sample_text(tok)
    explain = explainers.get_explainer(name, model, tok, CFG)

    def run(seed):
        set_seed(seed)
        return np.asarray(explain(text).raw_scores)

    model.eval()
    assert np.allclose(run(42), run(42))
    if name in ("grad_input", "lrp", "attention_rollout"):
        assert np.allclose(run(42), run(123)), "in eval i metodi deterministici non dipendono dal seed"
    model.train()
    try:
        a, b, c = run(42), run(42), run(123)
        assert np.allclose(a, b), "stesso seed, stessa spiegazione anche con dropout"
        assert not np.allclose(a, c), "seed diversi con dropout devono dare spiegazioni diverse"
    finally:
        model.eval()


def test_consistency_is_one_without_dropout_and_lower_with_it(tiny):
    model, tok = _load("tiny-bert")
    text = _sample_text(tok)
    explain = explainers.get_explainer("grad_input", model, tok, CFG)
    model.eval()
    r = metrics.consistency_for_text(explain, text, set_seed=set_seed)
    assert r["consistency"] == pytest.approx(1.0)
    model.train()
    try:
        r = metrics.consistency_for_text(explain, text, set_seed=set_seed)
    finally:
        model.eval()
    assert r["consistency"] < 1.0


# =============================================================================
# Pipeline completa e ripresa
# =============================================================================
def test_report_end_to_end_and_resume(tiny, tmp_path):
    import report
    out = tmp_path / "run"
    argv = ["--models", *KEYS, "--sample", "6", "--max-length", str(MAXLEN), "--flow-max-length", "32",
            "--lime-samples", "20", "--batch-size", "8", "--no-amp", "--device", "cpu", "--out-dir", str(out),
            "--local-model", *[f"{k}={v}" for k, v in tiny.items()]]
    assert report.main(argv) == 0
    for metric in ("robustness", "consistency", "contrastivity"):
        df = __import__("pandas").read_csv(out / f"{metric}_table.csv", index_col=0)
        assert set(df.index) == set(report.EXPLAINERS), metric
        assert set(df.columns) == set(KEYS), metric
        assert df.notna().all().all(), f"{metric}: valori mancanti\n{df}"
    for key in KEYS:
        for ex in report.EXPLAINERS:
            s = json.loads((out / key / ex / "summary.json").read_text())
            assert s["n_eval_errors"] == 0, (key, ex)
            assert s["robustness"]["n_valid"] == 6, (key, ex, s["robustness"])
            assert s["consistency"]["n_total"] == 6 and s["consistency"]["n_errors"] == 0, (key, ex)
            assert s["contrastivity"]["n_pos"] == 3 and s["contrastivity"]["n_neg"] == 3
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["n_texts"] == 6 and meta["config"]["max_length"] == MAXLEN

    # ripresa: un secondo lancio non ricalcola nulla
    sizes = {p: len(read_jsonl(p)) for p in out.rglob("*.jsonl")}
    assert report.main(argv) == 0
    assert {p: len(read_jsonl(p)) for p in out.rglob("*.jsonl")} == sizes

    # interruzione simulata: si cancellano le ultime righe di un file, il rilancio le rifà
    p = out / "tiny-bert" / "lrp" / "train.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:3]) + "\n" + lines[3][:10])  # ultima riga troncata
    assert report.main(argv) == 0
    assert len({r["i"] for r in read_jsonl(p)}) == 6


def test_parallel_run_gives_same_tables_as_sequential(tiny, tmp_path):
    """Il launcher parallelo (shard su più processi) deve dare gli stessi numeri del run unico."""
    import subprocess
    import sys
    import pandas as pd
    import report
    common = ["--sample", "6", "--max-length", str(MAXLEN), "--flow-max-length", "32", "--lime-samples", "20",
              "--batch-size", "8", "--no-amp", "--local-model", *[f"{k}={v}" for k, v in tiny.items()]]
    seq, par = tmp_path / "seq", tmp_path / "par"
    assert report.main(["--models", *KEYS, "--device", "cpu", "--threads", "1", "--out-dir", str(seq)] + common) == 0
    root = Path(report.__file__).resolve().parent
    cmd = [sys.executable, str(root / "parallel_run.py"), "--models", *KEYS, "--out-dir", str(par),
           "--gpu-lanes", "2", "--flow-procs", "2", "--poll", "30"] + common
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    for metric in ("robustness", "contrastivity"):
        a = pd.read_csv(seq / f"{metric}_table.csv", index_col=0).sort_index()
        b = pd.read_csv(par / f"{metric}_table.csv", index_col=0).reindex(index=a.index, columns=a.columns)
        assert np.allclose(a.values, b.values, rtol=1e-5, atol=1e-7), f"{metric}\n{a}\n{b}"
    a = pd.read_csv(seq / "consistency_table.csv", index_col=0).sort_index()
    b = pd.read_csv(par / "consistency_table.csv", index_col=0).reindex(index=a.index, columns=a.columns)
    to_num = lambda df: df.apply(lambda col: col.map(lambda v: float(v.split("±")[0])))
    assert np.allclose(to_num(a).values, to_num(b).values, atol=2e-4), f"{a}\n{b}"
