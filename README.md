# SCAPE: Semantic Complexity & Analysis Package

SCAPE estimates **semantic complexity** of text using encoder attention patterns. 
**SCAPE-T** is its encoder-only transformer variant.

The repo contains:

* **`scape.scape_t`**: the core SCAPE-T scorer (`scape_t`) and defaults
* **`scape.dataset_builder.py`**: optional dataset builders for semantic complexity evaluation
* **`scape.datasets`**: bundled evaluation datasets, importable as Python vars
* **Evaluation utilities**: results table + ranking helper
* **Ablation utilities**: submetric and per-layer ablations
* **Tuning utilities**: simple evolutionary hyperparameter search
* **Key tables**: tables produced by the above

Python ≥ **3.9** recommended.

---

## Quick install (Colab or local)

**Colab / Jupyter (bang commands):**

```bash
!pip install -U pip setuptools wheel
!pip install git+https://github.com/MaxKho/SCAPE-T.git
!python -m spacy download en_core_web_sm
```

**Terminal / local:**

```bash
pip install -U pip setuptools wheel
pip install git+https://github.com/MaxKho/SCAPE-T.git
python -m spacy download en_core_web_sm
```

> Notes
> • First run will download Hugging Face models (e.g., GPT-2 for eval helpers).
> • If you have a GPU, SCAPE-T and GPT-2 will use it automatically if CUDA is available.

---

## TL;DR usage

### 1) Score some text (SCAPE-T demo)

```python
from scape.scape_t.core import scape_t

texts = ["The individualistic yet universalist nature of existentialist analysis paradoxically betrays the inherent contradictions within this mode of thought.",
         "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making.",
         "In the beginning of years, when the world was so new and all, and the Animals were just beginning to work for Man, there was a Camel, and he lived in the middle of a Howling Desert because he did not want to work"]

scores = scape_t(texts)  # returns {text: score}
for t, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
    print(f"SENTENCE: {t}\nSCORE: {s:7.3f}")
```

*Script:* `scripts/demo.py`

---

### 2) Build the evaluation table (pairwise accuracy) and term ranking (abstractness set)

```python
from scape.scape_t.eval import build_results_table, ranking
display(build_results_table())
display(ranking())
```

*Script:* `scripts/eval.py`

---

### 3) Perform ablations (submetrics + per-layer)

```python
from scape.scape_t.ablations import submetric_ablation_dev, submetric_ablation_abstractness, layer_ablation_dev

b, df = submetric_ablation_dev(); print(f"\nDev baseline: {b:.2f}"); display(df)
b2, df2 = submetric_ablation_abstractness(); print(f"\nAbstractness baseline: {b2:.2f}"); display(df2)
layer_df, _ = layer_ablation_dev(); print("\nLayer ablations (Dev):"); display(layer_df)
```

*Script:* `scripts/ablations.py`

---

### 4) Tune the parameters (evolutionary search)

```python
from scape.scape_t.tuning import evolutionary_search
from scape import datasets

dev_corpus = {**{s:0 for s in datasets.Dev_simple}, **{s:1 for s in datasets.Dev_complex}}
best, history = evolutionary_search(dev_corpus, datasets.Dev_simple, datasets.Dev_complex, n_iter=100, n_offspring=10, scale=0.3)
print(f"Best ACC: {best['acc']:.4f}\nBest params: {best['params']}")
```

*Script:* `scripts/tune.py`

---

## What’s in the package?

### Core scorer

* `scape.scape_t.core.scape_t(text_or_iterable, p=DEFAULT_PARAMS, layer_abl=None, only_layer=None)`

  * Accepts a single string, or any container of strings (list/tuple/set), or a dict `{text -> class}`.
  * Returns a dict `{text -> score}` (for single string, returns the score for that string).
  * Optional per-layer ablation: `layer_abl` (remove layer *li*), `only_layer` (use only layer *li*).

### Defaults & resources

* `scape.scape_t.resources.DEFAULT_PARAMS` – default hyperparameters (dict)
* Other internal tokenization data (stopwords, punctuation maps, etc.)

### Bundled datasets (ready to import)

* Module: `scape.datasets`
* Exposed variables (lists/dicts):

  * `Dev_simple`, `Dev_complex`
  * `Test_simple`, `Test_complex`
  * `WLC_Test_simple`, `WLC_Test_complex`
  * `GAU_simple`
  * `Abstractness_Test` (dict: text/term → class id)
* Utilities:

  * `scape.datasets.list_datasets()`
  * `scape.datasets.as_dict()`

### Evaluation helpers

* `scape.scape_t.eval.build_results_table(datasets=None, params=None, device=None) -> pd.DataFrame`

  * Computes pairwise accuracies for several metrics on `WLC_Test` and `Test`,
    plus Abstractness (Class 4 vs 1 & 2).
  * Uses GPT-2 perplexity, lexicon scores, MDL bits/byte, spaCy syntax, SCAPE-T, etc.
* `scape.scape_t.eval.ranking(corpus=None, params=None, no_classes=4, datasets=None) -> pd.DataFrame`

  * Ranks a labelled corpus by SCAPE-T score and derives a naive “Predicted” class by cumulative counts.
  * Defaults to `datasets["Abstractness_Test"]`.

### Ablations

* `scape.scape_t.ablations.submetric_ablation_dev(params=None)`
* `scape.scape_t.ablations.submetric_ablation_abstractness(params=None)`
* `scape.scape_t.ablations.layer_ablation_dev(params=None)`

  * The layer ablation also **draws a color-box plot** (saved as `layer_importance_boxes.png`).

### Tuning

* `scape.scape_t.tuning.evolutionary_search(corpus, simple_sents, complex_sents, n_iter=50, n_offspring=5, scale=0.3)`

  * Simple evolutionary search over the hyperparameter space using dev pairwise accuracy as the objective.
 
### Advanced: (Re)build datasets from source

If you need to re-scrape/regenerate the datasets instead of using the bundled ones, use the builder utilities in `scape.dataset_builder`:

```python
from scape.dataset_builder import (
    get_dev_splits, get_test_splits, get_gau_split, get_all_datasets
)

Dev_simple, Dev_complex = get_dev_splits(n=200)
Test_simple, Test_complex, WLC_Test_simple, WLC_Test_complex = get_test_splits(n_len_ctrl=100)
GAU_simple = get_gau_split()
data = get_all_datasets()
```

---

## Performance tips

* **GPU**: If available, PyTorch/Transformers will use `cuda` automatically inside helpers; you can also pass `device` to `build_results_table`.
* **Avoid re-loading models** in tight loops; the package caches GPT-2, spaCy, and lexicon data across calls.
* **Batching**: `scape_t` internally batches over sentences; passing a list/dict is more efficient than calling it one sentence at a time.

---

## Data & licensing notes

* Bundled datasets are provided for evaluation convenience.
* If you rebuild datasets yourself (e.g., via scraping Gutenberg/Wikipedia), follow the respective sites’ terms of use.
* For demos/examples, prefer public-domain or your own text.

---

## Repo layout (source)

```
src/
  scape/
    __init__.py
    datasets/                  # data files (txt/jsonl/csv) importable via scape.datasets
      Dev_simple.txt
      Dev_complex.txt
      ...
      abstractness.csv
    dataset_builder.py         # (optional) original builders; not used at runtime
    scape_t/
      __init__.py
      core.py                  # scape_t, get_model, get_tokeniser, device, etc.
      resources.py             # DEFAULT_PARAMS, tokenization resources
      eval.py                  # build_results_table, ranking
      ablations.py             # submetric & per-layer ablations
      tuning.py                # evolutionary_search, mutate
```

---

## Citation

If you use SCAPE-T in academic work, please cite the repository and include a link to the code.
