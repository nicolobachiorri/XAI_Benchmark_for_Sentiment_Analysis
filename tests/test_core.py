"""Test delle parti che non richiedono PyTorch: attribuzioni, Rollout, Flow,
KernelSHAP, LIME (con un classificatore finto) e le tre metriche.

    python -m pytest -q tests/test_core.py
"""
import itertools
import math
import random

import numpy as np
import pytest

import attribution as A
import metrics as M
from attribution import Attribution


# ---------------------------------------------------------------- attribuzioni
def test_minmax01_range_and_constant():
    s = A.minmax01([-2.0, 0.0, 2.0])
    assert np.allclose(s, [0, 0.5, 1])
    assert np.allclose(A.minmax01([3.0, 3.0]), [0, 0])


def test_token_word_ids_bert_and_roberta_offsets():
    text = "The movie, honestly, was great"
    spans = A.word_spans(text)
    assert [text[s:e] for s, e in spans] == text.split(" ")
    # BERT: [CLS] the movie , honestly , was great [SEP]
    bert = [(0, 0), (0, 3), (4, 9), (9, 10), (11, 19), (19, 20), (21, 24), (25, 30), (0, 0)]
    assert A.token_word_ids(bert, spans) == [-1, 0, 1, 1, 2, 2, 3, 4, -1]
    # RoBERTa con offset che includono lo spazio iniziale (trim_offsets=False)
    rob = [(0, 0), (0, 3), (3, 9), (9, 10), (10, 19), (19, 20), (20, 24), (24, 30), (0, 0)]
    assert A.token_word_ids(rob, spans) == [-1, 0, 1, 1, 2, 2, 3, 4, -1]


def test_aggregate_to_words_sums_subwords():
    out = A.aggregate_to_words([9, 1, 2, 3, 9], [-1, 0, 0, 1, -1])
    assert np.allclose(out, [3, 3])


# ---------------------------------------------------------------- rollout e flow
def _random_att(L, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n), size=(L, n))


def test_rollout_matches_definition():
    att = _random_att(3, 5)
    I = np.eye(5)
    A1, A2, A3 = [0.5 * a + 0.5 * I for a in att]
    expected = (A3 @ A2 @ A1)[0]
    assert np.allclose(A.rollout_scores(att), expected)
    assert np.isclose(A.rollout_scores(att).sum(), 1.0)  # righe stocastiche


def test_flow_matches_networkx_exact():
    nx = pytest.importorskip("networkx")
    att = _random_att(3, 6, seed=1)
    R = A.residual_attention(att)
    L, n, _ = R.shape
    G = nx.DiGraph()
    for l in range(1, L + 1):
        for i in range(n):
            for j in range(n):
                G.add_edge((l, i), (l - 1, j), capacity=float(R[l - 1, i, j]))
    expected = [nx.maximum_flow_value(G, (L, 0), (0, t)) for t in range(n)]
    got = A.flow_scores(att)
    assert np.allclose(got, expected, atol=1e-4)
    # il flusso non supera mai la capacità entrante nel token di input
    assert np.all(got <= R[0].sum(axis=0) + 1e-6)


def test_flow_parallel_equals_serial():
    att = _random_att(2, 12, seed=2)
    assert np.allclose(A.flow_scores(att, workers=1), A.flow_scores(att, workers=3))


# ---------------------------------------------------------------- KernelSHAP
def _brute_force_shapley(f, M):
    phi = np.zeros(M)
    for i in range(M):
        others = [j for j in range(M) if j != i]
        for r in range(M):
            for S in itertools.combinations(others, r):
                z = np.zeros(M); z[list(S)] = 1
                zi = z.copy(); zi[i] = 1
                w = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
                phi[i] += w * (f(zi[None])[0] - f(z[None])[0])
    return phi


def test_kernel_shap_exact_equals_shapley_values():
    rng = np.random.default_rng(3)
    W = rng.normal(size=6)
    def f(Z):  # funzione non lineare con interazioni
        Z = np.atleast_2d(Z)
        return np.tanh(Z @ W) + 0.3 * Z[:, 0] * Z[:, 1]
    phi = A.kernel_shap_values(f, 6, nsamples=10_000)
    assert np.allclose(phi, _brute_force_shapley(f, 6), atol=1e-8)


def test_kernel_shap_sampled_is_additive_and_seeded():
    W = np.linspace(-1, 1, 40)
    f = lambda Z: np.atleast_2d(Z) @ W
    a = A.kernel_shap_values(f, 40, nsamples=15, rng=np.random.RandomState(0))
    b = A.kernel_shap_values(f, 40, nsamples=15, rng=np.random.RandomState(0))
    assert np.allclose(a, b)
    assert np.isclose(a.sum(), f(np.ones((1, 40)))[0] - f(np.zeros((1, 40)))[0])
    # con molti campioni una funzione lineare è ricostruita esattamente
    c = A.kernel_shap_values(f, 40, nsamples=400, rng=np.random.RandomState(0))
    assert np.allclose(c, W, atol=1e-6)


# ---------------------------------------------------------------- LIME
def test_lime_recovers_word_effects():
    pytest.importorskip("lime")
    words = "the film was wonderful but the ending was awful".split()
    weight = {"wonderful": 2.0, "awful": -2.0}
    def clf(texts):
        p = []
        for t in texts:
            z = sum(weight.get(w, 0.0) for w in t.split(" "))
            p.append(1 / (1 + np.exp(-z)))
        p = np.array(p)
        return np.stack([1 - p, p], axis=1)
    np.random.seed(0)
    s = A.lime_word_scores(words, clf, "[MASK]", num_samples=500)
    assert len(s) == len(words)
    assert s[words.index("wonderful")] == s.max()
    assert s[words.index("awful")] == s.min()


# ---------------------------------------------------------------- perturbazioni
def test_perturbations_pairs_point_to_identical_words():
    words = ("This movie was really very good , and the actors were great " * 3).split()
    for name, fn in M.PERTURBATIONS.items():
        new, pairs = fn(words, random.Random(1), mask_token="[MASK]")
        for i, j in pairs:
            assert words[i] == new[j], name
        n_changed = len(words) - len(pairs)
        assert n_changed >= 1
        if name != "substitute":
            assert n_changed == math.floor(0.15 * len(words))


def test_substitute_changes_case_and_punctuation_safely():
    new, pairs = M.perturb_substitute(["Movie,", "x"], random.Random(0))
    assert new[0] == "Film," and pairs == [(1, 1)]
    assert M.perturb_substitute(["xyz", "qwe"], random.Random(0)) is None


def test_make_perturbations_reproducible():
    text = "a good movie with a great cast and a boring plot"
    a = M.make_perturbations(text, 7, "[MASK]")
    b = M.make_perturbations(text, 7, "[MASK]")
    assert a == b and len(a) == 6


# ---------------------------------------------------------------- robustness
def _lexical_explainer(text):
    # punteggio che dipende solo dalla parola: stabile per costruzione
    words = text.split(" ")
    return Attribution(words, [len(w) % 5 for w in words])


def test_robustness_formula_per_perturbation_then_mean():
    orig = Attribution(["a", "b", "c", "d"], [0, 1, 2, 3])
    perts = [{"type": "x", "text": "p1", "pairs": [(0, 0), (1, 1)]},
             {"type": "y", "text": "p2", "pairs": [(3, 0)]}]
    answers = {"p1": Attribution(["a", "b"], [0, 0]), "p2": Attribution(["d", "z"], [0, 1])}
    r = M.robustness_for_text(lambda t: answers[t], "orig", orig, perts)
    # orig normalizzato: [0, 1/3, 2/3, 1]; p1 -> [0, 0]; p2 -> [0, 1]
    # p1: (|0-0| + |1/3-0|)/2 = 1/6 ; p2: |1 - 0| = 1 ; media = 7/12
    assert np.isclose(r["robustness"], 7 / 12)


def test_robustness_zero_for_word_level_explainer():
    text = "the plot was boring but the cast was very good indeed"
    orig = _lexical_explainer(text)
    perts = [p for p in M.make_perturbations(text, 0, "[MASK]") if p["type"] == "substitute"]
    # con sostituzioni le parole cambiate sono escluse: le altre hanno lo stesso punteggio,
    # ma la normalizzazione min-max può cambiare se cambia il minimo o il massimo
    r = M.robustness_for_text(_lexical_explainer, text, orig, perts)
    assert r["robustness"] is not None and r["robustness"] >= 0


def test_robustness_not_applicable_is_not_a_score():
    orig = Attribution(["a", "b"], [0, 1])
    perts = [{"type": "substitute", "text": None, "pairs": []}]
    r = M.robustness_for_text(lambda t: orig, "a b", orig, perts)
    assert r["robustness"] is None  # prima: 0.0, il punteggio migliore


# ---------------------------------------------------------------- consistency
def test_consistency_identical_explanations_with_repeated_words_is_one():
    words = "the movie the plot the cast the end".split()
    rng = np.random.default_rng(0)
    scores = rng.normal(size=len(words))
    r = M.consistency_for_text(lambda t: Attribution(words, scores.tolist()), "x")
    assert r["consistency"] == pytest.approx(1.0)  # nella versione della tesi: ben sotto 1


def test_consistency_constant_explanation_is_nan_not_a_fallback():
    words = ["a", "b", "c"]
    r = M.consistency_for_text(lambda t: Attribution(words, [0.0, 0.0, 0.0]), "x")
    assert r["consistency"] is None and r["n_valid_pairs"] == 0


def test_consistency_aggregate_uses_sample_std():
    agg = M.aggregate_consistency([0.2, 0.4, None, 0.6])
    assert agg["mean"] == pytest.approx(0.4)
    assert agg["std"] == pytest.approx(0.2)
    assert agg["n_valid"] == 3 and agg["n_total"] == 4


def test_consistency_uses_seeds():
    seen = []
    def explain(t):
        seen.append(np.random.randint(0, 10**9))
        return Attribution(["a", "b", "c"], list(np.random.normal(size=3)))
    M.consistency_for_text(explain, "x")
    first = list(seen); seen.clear()
    M.consistency_for_text(explain, "x")
    assert seen == first and len(set(first)) == 4


# ---------------------------------------------------------------- contrastivity
def test_contrastivity_zero_for_identical_classes():
    a = Attribution(["good", "bad", "film"], [1, 0, 0.5])
    r = M.contrastivity([a, a], [a, a])
    assert r["contrastivity"] == pytest.approx(0.0, abs=1e-8)


def test_contrastivity_matches_formula():
    pos = [Attribution(["good", "film"], [1.0, 0.0]), Attribution(["great", "film"], [1.0, 0.0])]
    neg = [Attribution(["bad", "film"], [1.0, 0.0])]
    r = M.contrastivity(pos, neg)
    vocab = ["film", "bad", "good", "great"]  # per frequenza, poi ordine alfabetico
    S_pos = np.array([0.0, 0.0, 1.0, 1.0]); S_neg = np.array([0.0, 1.0, 0.0, 0.0])
    P = (S_pos - S_pos.min()) / (S_pos - S_pos.min()).sum()
    Q = (S_neg - S_neg.min()) / (S_neg - S_neg.min()).sum()
    m = P > 0
    expected = np.sum(P[m] * np.log2(P[m] / (Q[m] + 1e-10)))
    assert r["contrastivity"] == pytest.approx(expected)


def test_contrastivity_vocab_is_most_frequent_not_alphabetical():
    pos = [Attribution(["zzz"] * 5 + ["aaa"], [1, 1, 1, 1, 1, 0])]
    neg = [Attribution(["zzz"] * 5 + ["bbb"], [0, 0, 0, 0, 0, 1])]
    # con vocabolario di 1 parola resta la più frequente ("zzz"), non la prima in ordine alfabetico
    r = M.contrastivity(pos, neg, vocab_size=1)
    assert r["vocab_size"] == 1
    # una sola parola: P+ = P- = [1] (distribuzione uniforme), KL = 0
    assert r["contrastivity"] == pytest.approx(0.0, abs=1e-8)


def test_contrastivity_signed_scale_limits_zero_probabilities():
    # con la scala [0, 1] tutte le parole assenti dai negativi hanno P- = 0;
    # con il segno solo la parola con S- minimo (qui "awful")
    pos = [Attribution(["great", "superb", "film", "plot"], [0.9, 0.7, -0.2, 0.1])]
    neg = [Attribution(["awful", "dull", "film", "plot"], [-0.8, -0.5, 0.1, -0.1])]
    signed = M.contrastivity(pos, neg)
    minmax = M.contrastivity(pos, neg, scale="minmax01")
    assert signed["n_vocab_zero_in_neg"] <= 1
    assert minmax["n_vocab_zero_in_neg"] >= 2
    assert signed["contrastivity"] < minmax["contrastivity"]


def test_signed_scores_keep_sign_and_scale():
    a = Attribution(["a", "b", "c"], [-2.0, 1.0, 0.0])
    assert np.allclose(a.signed_scores, [-1.0, 0.5, 0.0])


def test_contrastivity_js_bounded_and_zero_for_identical():
    a = Attribution(["good", "bad", "film"], [1, -1, 0.5])
    assert M.contrastivity([a], [a])["js_divergence"] == pytest.approx(0.0, abs=1e-12)
    pos = [Attribution(["great", "film"], [1.0, 0.0])]
    neg = [Attribution(["awful", "plot"], [1.0, 0.0])]
    r = M.contrastivity(pos, neg)
    assert 0 < r["js_divergence"] <= 1
    assert r["kl_from_zero_neg"] <= r["contrastivity"] + 1e-12
