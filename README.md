# SCAPE: Semantic Complexity & Analysis Package

SCAPE estimates **semantic complexity** of text using encoder attention patterns. 
**SCAPE-T** is its encoder-only transformer variant.

The repo contains:

* **`scape.scape_t`**: the core SCAPE-T scorer (`scape_t`) and defaults
* **`scape.dataset_builder.py`**: optional dataset builders for semantic complexity evaluation
* **`scape.datasets`**: bundled evaluation datasets, importable as Python vars
  > **Note:** In this repository, **`Dev_simple`** and **`GAU_simple`** are **not the same as in the SCAPE-T paper**. They have been **replaced by AI-generated text** for licensing/jurisdiction compliance. See **“Data & Licensing (UK)”** below for details.
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
!pip install git+https://github.com/MaxKho/SCAPE.git
!python -m spacy download en_core_web_sm
```

**Terminal / local:**

```bash
pip install -U pip setuptools wheel
pip install git+https://github.com/MaxKho/SCAPE.git
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


### 2) Build the evaluation table (pairwise accuracy) and term ranking (abstractness set)

```python
from scape.scape_t.eval import build_results_table, ranking
display(build_results_table())
display(ranking())
```

*Script:* `scripts/eval.py`


### 3) Perform ablations (submetrics + per-layer)

```python
from scape.scape_t.ablations import submetric_ablation_dev, submetric_ablation_abstractness, layer_ablation_dev

b, df = submetric_ablation_dev(); print(f"\nDev baseline: {b:.2f}"); display(df)
b2, df2 = submetric_ablation_abstractness(); print(f"\nAbstractness baseline: {b2:.2f}"); display(df2)
layer_df, _ = layer_ablation_dev(); print("\nLayer ablations (Dev):"); display(layer_df)
```

*Script:* `scripts/ablations.py`



### 4) Tune the parameters (evolutionary search)
> **Note:** In this repository, **`datasets.Dev_simple`** is **AI-generated** (not the same as in the SCAPE-T paper) for licensing reasons. See **“Data & Licensing (UK)”** for details. Results may differ slightly from the paper where the original `dev_simple` was used.

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

  * `Dev_simple` *(AI-generated replacement; not the same as in the SCAPE-T paper — see “Data & Licensing (UK)”)*
  * `Dev_complex`
  * `Test_simple`
  * `Test_complex`
  * `WLC_Test_simple`
  * `WLC_Test_complex`
  * `GAU_simple` *(AI-generated replacement; not the same as in the SCAPE-T paper — see “Data & Licensing (UK)”)*
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
    > **Note:** If you pass `datasets.Dev_simple`/`GAU_simple` from this repo, remember these are **AI-generated replacements** (different from the SCAPE-T paper). See **“Data & Licensing (UK)”**.
 
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

## Repo layout (source)

```
src/
  scape/
    __init__.py
    datasets/                  # data files (txt/jsonl/csv) importable via scape.datasets
      abstractness.csv
      dev_complex.txt
      dev_simple.txt           # NOTE: AI-generated replacement (not the same as in the SCAPE-T paper) — see Data & Licensing
      gau_simple.txt           # NOTE: AI-generated replacement (not the same as in the SCAPE-T paper) — see Data & Licensing
      ...
      wlc_test_simple.txt
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

---

# Data & Licensing (UK)

**Jurisdiction.** This repository is maintained from the **United Kingdom**, where literary works are protected for **life + 70 years** (UK law). Public-domain (PD) status for non-CC texts in this repo is assessed under **UK law**.

**Access dates.** All source texts referenced below were accessed on **18/08/2025**.

---

## What this dataset contains

* **Wikipedia sentences** (bulk reuse; hundreds–thousands) used as the *complex* group in evaluations.
* **Public-domain children’s stories** (Project Gutenberg, UK-PD) used as the *simple* group.
* **Classic philosophy extracts** (Project Gutenberg, UK-PD) used as *complex* development material.
* **Two synthetic subsets (ChatGPT)** used for compliance in place of content I may not redistribute:

  * `scape/datasets/dev_simple.txt` — **200 AI-generated sentences** in the style of MCTest (no MCTest text included).
    > **Note:** This **replaces** the `dev_simple` group used in the **SCAPE-T paper**. See the licensing rationale in **“Synthetic subsets (details & licence)”** below.
  * `scape/datasets/gau_simple.txt` — **200 AI-generated French sentences** in the style of a children’s story (no *Le Petit Prince* text included).
    > **Note:** This **replaces** the `gau_simple` group used in the **SCAPE-T paper**. See the licensing rationale in **“Synthetic subsets (details & licence)”** below.

---

## Licences & redistribution

### Wikipedia

Portions of this dataset **reproduce Wikipedia text** and are therefore licensed under **CC BY-SA 4.0**. Reuse requires:

* **Attribution** (see TASL below),
* A **link to the licence**,
* **Indication of changes** (see Indication of Changes (pre-processing) below), and
* **Share-alike** (derivatives must also be under **CC BY-SA 4.0**).

**Licence link (human-readable):** [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)
**Legal code:** [https://creativecommons.org/licenses/by-sa/4.0/legalcode](https://creativecommons.org/licenses/by-sa/4.0/legalcode)

> **Share-alike reminder:** Any files in this repo that contain Wikipedia text are re-licensed under **CC BY-SA 4.0**. If you remix those files, your derivatives must also be **CC BY-SA 4.0** and retain attribution and change notes.

### Project Gutenberg (US)

Texts included here are **public domain in the UK**. Keep **source URLs** and do not imply endorsement by Project Gutenberg (registered trademark). If you redistribute their **editions** verbatim, retain their notices.

### Project Gutenberg Australia (PG-AU)

*Le Petit Prince* is **public domain in the UK/EU** but **not** in the **United States** until 2039. Because GitHub is US-based, this repository **does not redistribute** the PG-AU text. Instead, I provide a **synthetic** French subset (see below) and code that can fetch PG-AU content locally for research use where lawful.

### Synthetic subsets (ChatGPT)

* Generated with ChatGPT (OpenAI). Under OpenAI’s Terms, **you own the outputs** to the extent permitted by law.
* To avoid licence conflicts and keep reuse simple, both synthetic subsets are released under **CC0 1.0** (public-domain dedication).

  * **CC0 1.0:** [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)

Folders/files **not** containing Wikipedia text (e.g., the synthetic subsets) are **not** subject to CC BY-SA; they carry the licence stated here (CC0 1.0).

---

## Indication of changes (pre-processing)

For all source texts, I applied **non-substantive normalisation**:

* Sentence segmentation and filtering of very short sentences (≥10 words).
* Removal of bracketed references and boilerplate (e.g., `[\d+]`, editorial notes, headers/footers).
* Removal of inline numeric artefacts and digits stuck to words.
* Whitespace/punctuation normalisation.

No paraphrasing or semantic rewrites of source sentences were performed.

---

## TASL attribution for Wikipedia content

The dataset includes material from the following articles (accessed **20/08/2025**). Each entry provides **T**itle, **A**uthor, **S**ource, **L**icence, plus **Changes** as required by CC BY-SA.

1. **Title:** Artificial intelligence
   **Author:** Wikipedia contributors
   **Source:** [https://en.wikipedia.org/wiki/Artificial\_intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence)
   **Licence:** CC BY-SA 4.0 — [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)
   **Changes:** sentence segmentation; removal of references/boilerplate; light normalisation.

2. **Title:** Quantum field theory
   **Author:** Wikipedia contributors
   **Source:** [https://en.wikipedia.org/wiki/Quantum\_field\_theory](https://en.wikipedia.org/wiki/Quantum_field_theory)
   **Licence:** CC BY-SA 4.0 — [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)
   **Changes:** sentence segmentation; removal of references/boilerplate; light normalisation.

3. **Title:** Philosophy of mind
   **Author:** Wikipedia contributors
   **Source:** [https://en.wikipedia.org/wiki/Philosophy\_of\_mind](https://en.wikipedia.org/wiki/Philosophy_of_mind)
   **Licence:** CC BY-SA 4.0 — [https://creativecommons.org/licenses/by-sa/4.0/](https://creativecommons.org/licenses/by-sa/4.0/)
   **Changes:** sentence segmentation; removal of references/boilerplate; light normalisation.

> If you add more Wikipedia pages, append TASL entries here with title, URL, access date, licence link, and change notes.

---

## Public-domain sources (UK)

**Children’s stories (Project Gutenberg):**

* *The Three Little Pigs* — [https://www.gutenberg.org/cache/epub/18155/pg18155.txt](https://www.gutenberg.org/cache/epub/18155/pg18155.txt) — **Public Domain (UK)** — accessed 18/08/2025.
* *The Little Gingerbread Man* — [https://www.gutenberg.org/files/25877/25877.txt](https://www.gutenberg.org/files/25877/25877.txt) — **Public Domain (UK)** — accessed 18/08/2025.
* *The Children’s Hour (collection)* — [https://www.gutenberg.org/files/11592/11592-h/11592-h.htm](https://www.gutenberg.org/files/11592/11592-h/11592-h.htm) — **Public Domain (UK)** — accessed 18/08/2025.

**Classic philosophy (Project Gutenberg):**

* G.W\.F. Hegel — Project Gutenberg eBook #55108 — [https://www.gutenberg.org/ebooks/55108.txt.utf-8](https://www.gutenberg.org/ebooks/55108.txt.utf-8) — **Public Domain (UK)** — accessed 18/08/2025.
* Immanuel Kant — Project Gutenberg eBook #4280 — [https://www.gutenberg.org/ebooks/4280.txt.utf-8](https://www.gutenberg.org/ebooks/4280.txt.utf-8) — **Public Domain (UK)** — accessed 18/08/2025.

**Classic literature (PG-AU; not redistributed here):**

* *Le Petit Prince* — [https://gutenberg.net.au/ebooks03/0300771h.html](https://gutenberg.net.au/ebooks03/0300771h.html) — **Public Domain (UK)**; **not PD in the US** — accessed 18/08/2025.
  *Reason for omission:* GitHub is US-based; I therefore **do not host** the text and instead provide a **synthetic** French subset (below).

---

## Synthetic subsets (details & licence)

**`scape/datasets/dev_simple.txt` — MCTest-style synthetic (English).**

* **Why synthetic?** The MCTest dataset is distributed under a Microsoft Research licence that typically **prohibits redistribution**; therefore the original MCTest sentences are **not** included.
* **How created:** 200 **original** sentences generated with ChatGPT on 18/08/2025, prompted to resemble the **style** and **reading level** of MCTest (no quotes; no copying).
* **Licence:** **CC0 1.0** — [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)
* **Provenance note:** Under OpenAI’s Terms, outputs are owned by the user to the extent permitted by law.

**`scape/datasets/gau_simple.txt` — French children’s story synthetic.**

* **Why synthetic?** *Le Petit Prince* is PD in the UK but **not** in the US; to avoid US distribution issues on GitHub, the original text is **not** included.
* **How created:** 200 **original** French sentences generated with ChatGPT on 18/08/2025 in the **style** of a simple children’s story (no quotes; no copying).
* **Licence:** **CC0 1.0** — [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)
* **Provenance note:** As above.

> **Separation of licences:** Synthetic files (CC0) are kept separate from Wikipedia-derived files (CC BY-SA). Do not merge them into single files, or the merged files will need to be **CC BY-SA** due to share-alike.

---

## Code vs data licences

* **Wikipedia-derived data:** **CC BY-SA 4.0** (see licence links above).
* **Public-domain subsets:** **Public Domain (UK)** (sources listed).
* **Synthetic subsets (ChatGPT):** **CC0 1.0**.
* **Code: MIT License**; see the top‑level LICENSE file for full terms.

### Collection method & platform etiquette

Data was fetched from publicly available endpoints at modest rates for research. For bulk use of Wikipedia, Wikimedia recommends the **API or dumps** and a descriptive **User-Agent**; this does not affect the licences above but is good practice.

### How to attribute this dataset (downstream users)

When you use files containing Wikipedia text, include something like:

> “This dataset contains text from Wikipedia, available under the **CC BY-SA 4.0** licence. © Wikipedia contributors. Source article(s): *Artificial intelligence*, *Quantum field theory*, *Philosophy of mind* (accessed 18/08/2025). Changes: sentence segmentation; removal of references/boilerplate; light normalisation. Licence: [https://creativecommons.org/licenses/by-sa/4.0/.”](https://creativecommons.org/licenses/by-sa/4.0/.”)

For synthetic or UK-PD subsets, cite the relevant section above.
