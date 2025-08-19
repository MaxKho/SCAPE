"""
Evaluation utilities for SCAPE-T (baselines, table builder, and ranking).

This module benchmarks SCAPE-T against standard readability/complexity baselines
and provides two public entry points:

- `build_results_table(...)` – produce a DataFrame of pairwise accuracies for
  multiple metrics across the packaged splits:
    * WLC_Test (length-controlled)
    * Test (raw)
    * Abstractness_Test (class 4 vs classes 1&2, tie = 0.5)

- `ranking(...)` – score an annotated corpus and return a DataFrame ranked by
  SCAPE-T “Semantic Complexity”, with a derived “Predicted” class assignment
  that respects the class cardinalities in the gold labels.

Caching & performance:
    - GPT-2 weights/tokenizer are cached via @lru_cache (per device type).
    - The DHH lexicon TSV is fetched once per process via @lru_cache.
    - spaCy nlp object is initialised once and reused; pipeline runs in
      batches through nlp.pipe.
    - `build_results_table(device=...)` forwards the device **only** to GPT-2
      baselines (PPL & MDL). If `device` is None, we auto-pick CUDA when available.

Return shapes:
    - `build_results_table(...)` → pandas.DataFrame with index “Metric” and
      columns: “WLC_Test”, “Test”, “Abstractness_Test” (values in [0,1]).
    - `ranking(...)` → pandas.DataFrame with columns:
      “Text”, “Semantic Complexity”, “Class”, “Token Length”, “Predicted”.

Usage examples:
    #1
    from scape.eval import build_results_table, ranking
    tbl = build_results_table()                         # uses packaged datasets
    tbl.loc[["SCAPE-T", "GPT-2 perplexity"]]

    #2
    rank_df = ranking()                                 # defaults to Abstractness_Test
     rank_df.head(10)
"""

# ---- imports ----
import io, re, math, requests
from typing import Dict, List
import numpy as np
import pandas as pd
import torch, textstat, wordfreq, spacy
from functools import lru_cache
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scape.scape_t.core import scape_t
from scape.scape_t.resources import DEFAULT_PARAMS
from scape.dataset_builder import bert_token_length

# ---- cached resources (perf) ----
@lru_cache(maxsize=2)
def _get_gpt2(device_type: str = "cpu"):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    lm  = GPT2LMHeadModel.from_pretrained("gpt2")
    lm.to(torch.device(device_type)).eval()
    return tok, lm

@lru_cache(maxsize=1)
def _get_lexicon():
    url = "https://raw.githubusercontent.com/oliveralonzo/DHH-lexical-dataset/main/General%20Lexicon%20DHH%20Annotations.tsv"
    df = pd.read_csv(io.StringIO(requests.get(url).text), sep="\t", usecols=["word","average"])
    lex_map  = dict(zip(df["word"].str.lower(), df["average"]))
    lex_mean = float(df["average"].mean())
    return lex_map, lex_mean

_NLP = None
def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP

# ---- scoring and dataset retrieval helpers  ---- #
def pairwise_accuracy(score_dict, simple_sents, complex_sents):
    simple_scores, complex_scores = [score_dict[s] for s in simple_sents if s in score_dict], [score_dict[s] for s in complex_sents if s in score_dict]
    correct = sum(0.5 if cs == ss else float(cs > ss) for cs in complex_scores for ss in simple_scores)
    total = len(simple_scores) * len(complex_scores)
    return float("nan") if total == 0 else correct / total

def _get_default_datasets():
    """Return packaged datasets. Imported lazily to avoid import loops."""
    from scape import datasets
    return {
        "Dev_simple":         datasets.Dev_simple,
        "Dev_complex":        datasets.Dev_complex,
        "Test_simple":        datasets.Test_simple,
        "Test_complex":       datasets.Test_complex,
        "WLC_Test_simple":    datasets.WLC_Test_simple,
        "WLC_Test_complex":   datasets.WLC_Test_complex,
        "GAU_simple":         datasets.GAU_simple,
        "Abstractness_Test":  datasets.Abstractness_Test
    }

# ---- baseline scorers ----
def neg_flesch(s: str) -> float:
    # higher = harder
    return -textstat.flesch_reading_ease(s)

def token_len(s: str) -> int:
    return bert_token_length(s)

def ttr(s: str) -> float:
    w = re.findall(r"\w+", s.lower())
    return len(set(w)) / len(w) if w else 0.0

def neg_zipf(s: str) -> float:
    w = re.findall(r"[A-Za-z]+", s.lower())
    freqs = [wordfreq.zipf_frequency(t, "en") for t in w]
    return -(sum(freqs) / len(freqs) if freqs else 0.0)

@torch.no_grad()
def gpt2_ppl_list(ss: List[str], device: torch.device) -> List[float]:
    tok, lm = _get_gpt2("cuda" if device.type == "cuda" else "cpu")
    vals = []
    for s in ss:
        enc = tok(s, return_tensors="pt").to(device)
        out = lm(**enc, labels=enc["input_ids"])
        vals.append(torch.exp(out.loss).item())
    return vals

@torch.no_grad()
def mdl_bits_per_byte_list(ss: List[str], device: torch.device) -> List[float]:
    """Shannon code length estimate: total NLL (bits) / UTF-8 bytes."""
    tok, lm = _get_gpt2("cuda" if device.type == "cuda" else "cpu")
    out_vals = []
    for s in ss:
        enc = tok(s, return_tensors="pt").to(device)
        out = lm(**enc, labels=enc["input_ids"])
        total_nats = out.loss.item() * enc["input_ids"].numel()
        total_bits = total_nats / math.log(2.0)
        out_vals.append(total_bits / max(1, len(s.encode("utf-8"))))
    return out_vals

def lexicon_complexity_list(ss: List[str]) -> List[float]:
    lex_map, lex_mean = _get_lexicon()
    out = []
    for s in ss:
        toks = re.findall(r"[A-Za-z]+", s.lower())
        vals = [lex_map.get(t, lex_mean) for t in toks]
        out.append(sum(vals)/len(vals) if vals else lex_mean)
    return out

def spacy_mean_dep_depth(ss: List[str], batch_size: int = 512) -> List[float]:
    if not ss:
        return []
    nlp = _get_nlp()
    # allow long docs safely
    nlp.max_length = max(nlp.max_length, max(len(s) for s in ss) + 100)
    disable = [c for c in nlp.pipe_names if c != "parser"]
    scores = []
    for doc in nlp.pipe(ss, batch_size=batch_size, disable=disable):
        if not len(doc):
            scores.append(0.0); continue
        depths = [len(list(tok.ancestors)) for tok in doc]
        scores.append(sum(depths)/len(depths))
    return scores

def semantic_scores_for_list(ss: List[str], params: Dict) -> List[float]:
    score_map = scape_t(ss, p=params)
    return [score_map.get(s, float("nan")) for s in ss]

def _acc_4_vs_12_from_list(scores: List[float], cats: List[int]) -> float:
    arr = np.asarray(scores, dtype=float)
    cats = np.asarray(cats)
    mask4  = (cats == 4) & ~np.isnan(arr)
    mask12 = ((cats == 1) | (cats == 2)) & ~np.isnan(arr)
    s4, s12 = arr[mask4], arr[mask12]
    if s4.size == 0 or s12.size == 0:
        return float("nan")
    diff  = s4[:, None] - s12[None, :]
    wins  = (diff > 0).sum()
    ties  = (diff == 0).sum()
    total = diff.size
    return (wins + 0.5 * ties) / total

# ---- main API ----
def build_results_table(datasets: Dict[str, List[str]] | None = None,
                        params: Dict = None,
                        device: torch.device | None = None) -> pd.DataFrame:
    """
    Build the results table with pairwise accuracy on:
      - WLC_Test (length-controlled)
      - Test     (raw)
      - Abstractness_Test (Class 4 vs 1&2)
    Includes: Flesch–Kincaid, Token length, TTR, Zipf frequency,
              GPT-2 perplexity, Lexicon complexity, MDL (bits/byte),
              SCAPE_T, Mean dep-tree depth.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if datasets is None:
        datasets = _get_default_datasets()
    if params is None:
        params = DEFAULT_PARAMS

    Test_simple, Test_complex = datasets["Test_simple"], datasets["Test_complex"]
    WLC_Test_simple, WLC_Test_complex = datasets["WLC_Test_simple"], datasets["WLC_Test_complex"]

    items_STT = list(datasets["Abstractness_Test"].keys())
    cats_STT  = [datasets["Abstractness_Test"][s] for s in items_STT]
    sem_map_STT = scape_t(datasets["Abstractness_Test"], p=params)

    def fill_pairwise(simp: List[str], comp: List[str], scorer_fn) -> float:
        s_scores, c_scores = scorer_fn(simp), scorer_fn(comp)
        sd = dict(zip(simp, s_scores)); sd.update(zip(comp, c_scores))
        return pairwise_accuracy(sd, simp, comp)

    # One-pass specs
    specs = [
        ("Flesch–Kincaid hardness", lambda ss: [neg_flesch(s) for s in ss]),
        ("Token length",            lambda ss: [token_len(s) for s in ss]),
        ("Type–token ratio",        lambda ss: [ttr(s) for s in ss]),
        ("Zipf-freq hardness",      lambda ss: [neg_zipf(s) for s in ss]),
        ("GPT-2 perplexity",        lambda ss: gpt2_ppl_list(ss, device)),
        ("Lexicon complexity",      lambda ss: lexicon_complexity_list(ss)),
        ("MDL (bits/byte, gpt2)",   lambda ss: mdl_bits_per_byte_list(ss, device)),
        ("SCAPE-T",                 lambda ss: semantic_scores_for_list(ss, params)),
        ("Mean dep-tree depth",     lambda ss: spacy_mean_dep_depth(ss))
        ]

    splits = {"WLC_Test": (WLC_Test_simple, WLC_Test_complex),
              "Test":     (Test_simple, Test_complex)}

    results: Dict[str, Dict[str, float]] = {}

    # pairwise across WLC_Test and Test
    for name, fn in specs:
        inner = {}
        for split_name, (simp, comp) in splits.items():
            inner[split_name] = fill_pairwise(simp, comp, fn)
        results[name] = inner

    # Abstractness 4 vs 1&2
    for name, fn in specs:
        if name == "SCAPE-T":
            vals = [sem_map_STT.get(s, float("nan")) for s in items_STT]
        else:
            vals = fn(items_STT)
        results[name]["Abstractness_Test"] = _acc_4_vs_12_from_list(vals, cats_STT)

    # Build DataFrame with the same column names as (1)
    rows = [{
        "Metric": metric,
        "WLC_Test": cols.get("WLC_Test", float("nan")),
        "Test":     cols.get("Test",     float("nan")),
        "Abstractness_Test": cols.get("Abstractness_Test", float("nan")),
    } for metric, cols in results.items()]

    table = (pd.DataFrame(rows)
               .set_index("Metric")
               .sort_values(by="WLC_Test", ascending=False, na_position="last"))
    return table.round(3)

# ---- ranking utility ----
def ranking(corpus: Dict[str, int] | None = None,
            params: Dict = None,
            no_classes: int = 4,
            datasets: Dict[str, List[str]] | None = None) -> pd.DataFrame:
    """
    Compute summary statistics given semantic complexity scores on `corpus`.
    `corpus` must be a dict {text -> class_label in 1..no_classes}.
    Returns a DataFrame sorted by Semantic Complexity with a 'Predicted' class
    assigned by cumulative class-count boundaries down the ranking.
    """
    if datasets is None:
        datasets = _get_default_datasets()
    if corpus is None:
        corpus = datasets["Abstractness_Test"]
    if params is None:
        params = DEFAULT_PARAMS
    if not isinstance(corpus, dict):
        raise ValueError("`corpus` must be a dict {text -> class_label}.")

    texts   = list(corpus.keys())
    classes = [corpus[t] for t in texts]

    # score with scape_t
    scores  = semantic_scores_for_list(texts, params)
    lengths = [bert_token_length(s) for s in texts]

    df = (pd.DataFrame({
            "Text":                texts,
            "Semantic Complexity": scores,
            "Class":               classes,
            "Token Length":        lengths,
         })
         .sort_values(by="Semantic Complexity", ascending=False)
         .reset_index(drop=True))

    # Build class boundaries (from highest class down), then assign predicted
    counts = df["Class"].value_counts().to_dict()
    counts_by_label = [counts.get(c, 0) for c in range(1, no_classes + 1)]

    cum = 0
    boundaries = {}  # label -> cumulative count from the top segment
    for i in reversed(range(no_classes)):      # no_classes-1, ..., 0
        cum += counts_by_label[i]
        boundaries[i + 1] = cum

    # Predicted label by how many boundaries the row index has crossed
    idx = np.arange(len(df))
    df["Predicted"] = no_classes - sum((idx >= b).astype(int) for b in boundaries.values())

    return df
