"""
dataset_builder.py

Utilities for constructing, filtering, and sampling datasets used in semantic complexity
evaluation with SCAPE-T. This module provides functions to scrape and preprocess texts
from sources such as Project Gutenberg, Wikipedia, and MCTest, and to assemble balanced
evaluation splits.

Main components:
- Filtering helpers:
  * Tokenization- and length-based filtering of sentences
  * Cleaning of Wikipedia markup, Gutenberg headers/footers, and children’s story boilerplate

- Loading helpers:
  * Gutenberg philosophical works (Hegel, Kant)
  * Gutenberg children’s stories (fairy tales, Children’s Hour anthology)
  * Wikipedia technical articles (AI, QFT, philosophy of mind)
  * MCTest dataset via Hugging Face Datasets
  * Australian Gutenberg children's French-language children's book (Le Petit Prince)

- Dataset builders:
  * Greedy length-matching of simple/complex sentence pairs
  * Controlled sampling with average word-length and percentile-based constraints
  * Assembly of development, test, and special-purpose splits

- Public entry points:
  * `get_dev_splits()` – length-matched development data (MCTest vs. Hegel/Kant)
  * `get_test_splits()` – out-of-domain test data (children’s stories vs. Wikipedia),
    plus length-controlled subsets
  * `get_gau_split()` – auxiliary Australian Gutenberg data
  * `get_all_datasets()` – unified access to all curated splits, including abstractness test items

All returned datasets are tokenized and filtered for consistent downstream use in SCAPE-T
evaluation.
"""


# ---- imports ----
import re, io, math, random, requests
from collections import defaultdict
from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from datasets import load_dataset
from scape.scape_t.core import get_tokeniser, group_text, split_text
from scape.datasets import Abstractness_Test

# ---- initialisation ---
SEED = 1
tokeniser   = get_tokeniser()

# ---- filtering helpers ----
def bert_token_length(s: str) -> int:
    return len(tokeniser.tokenize(s))

def length_filter(text, min_length=10):
    sentences = group_text(split_text(text))
    return [s for s in (s.strip() for s in sentences) if len(s.split()) >= min_length]

def filter_gutenberg(sentences, min_tokens=10):
    seen, pairs = set(), []
    for s in sentences:
        if s in seen: continue
        seen.add(s)
        cnt = len(tokeniser.tokenize(s))
        if cnt >= min_tokens:
            pairs.append((cnt, s))
    return pairs

def filter_wiki(text):
    text = re.sub(r"\[\d+\]", "", text)  # Remove bracketed citations
    text = re.sub(r"(?<=[,.;:!?\"'])\s*\d{1,3}(?=\s|$)", "", text)  # Inline numeric artifacts
    text = re.sub(r"(?<=[a-zA-Z])\d{2,3}(?!\d)", "", text)  # Digits stuck to words
    return text

def filter_children_hour(sentences):
    # Remove any sentence that contains '//', the substring 'EBook', or a word of ≥3 consecutive uppercase letters
    def is_sentinel(s: str) -> bool:
        return (bool(re.search(r'\b[A-Z]{3,}\b', s)) or '//' in s or 'EBook' in s)
    return [s for s in sentences if not is_sentinel(s)]

# ---- loading helpers ----
def load_and_filter_gutenberg(url, skip_start=150, skip_end=200, min_tokens=10):
    r = requests.get(url); r.raise_for_status()
    txt = r.text
    start = re.search(r"\*+ START.*?\*+", txt)
    end   = re.search(r"\*+ END.*?\*+", txt)
    core = txt[start.end():end.start()] if start and end else txt
    core = re.sub(r"\s+", " ", core)
    sents = length_filter(core)
    sents = sents[skip_start:-skip_end]
    return filter_gutenberg(sents, min_tokens)

def load_and_filter_mctest(version):
    ds = load_dataset("sagnikrayc/mctest", version)
    texts = ds['train']['story'] + ds['validation']['story'] + ds['test']['story']
    joined = " ".join(t.replace("\\newline"," ") for t in texts)
    sents = length_filter(joined)
    return sents

def load_and_filter_wiki():
    topics = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Quantum_field_theory",
        "https://en.wikipedia.org/wiki/Philosophy_of_mind"
    ]
    all_text = ""
    for url in topics:
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        for p in soup.find_all("p"):
            paragraph = filter_wiki(p.get_text())
            all_text += paragraph + " "
    return length_filter(all_text)

def load_and_filter_children_hour():
    url = "https://www.gutenberg.org/files/11592/11592-h/11592-h.htm"
    r   = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    sents = []
    started = False

    for tag in soup.find_all(['h1','h2','h3','p']):
        txt = tag.get_text(strip=True)

        # 1) skip until the first real story heading is seen
        if not started:
            if tag.name in ('h1','h2','h3') and 'LITTLE RED RIDING HOOD' in txt.upper():
                started = True
            else:
                continue

        # 2) once started, break if the Gutenberg footer is hit
        if 'END OF THE PROJECT GUTENBERG EBOOK' in txt.upper():
            break

        # 3) if it's a heading, advance to the next tag
        if tag.name in ('h1','h2','h3'):
            continue

        # 4) else it's a <p> with story text
        #    skip any remaining boilerplate
        if 'PROJECT GUTENBERG' in txt:
            continue

        # 5) strip bracketed tags
        clean = re.sub(r"\[[^\]]*\]", "", txt)

        # 6) sentence-split
        sents.extend(length_filter(clean))

    return sents

def load_and_filter_children_stories():
    urls = [
      "https://www.gutenberg.org/cache/epub/18155/pg18155.txt",  # Three Little Pigs
      "https://www.gutenberg.org/files/25877/25877.txt"         # The Little Gingerbread Man
    ]
    all_sents = []
    for url in urls:
        txt = requests.get(url).text
        # cut out Gutenberg header/footer
        m1 = re.search(r"\*\*\* START OF .*?\*\*\*", txt)
        m2 = re.search(r"\*\*\* END OF .*?\*\*\*", txt)
        core = txt[m1.end():m2.start()] if m1 and m2 else txt
        core = re.sub(r"\s+", " ", core)
        core = re.sub(r"\[[^\]]*\]", "", core)
        all_sents.extend(length_filter(core))
    all_sents.extend(load_and_filter_children_hour())
    return filter_children_hour(all_sents)

def load_and_filter_gutenberg_au(url="https://gutenberg.net.au/ebooks03/0300771h.html", min_tokens=10):
    r = requests.get(url); r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # raw text, normalised whitespace
    txt = soup.get_text(separator=" ")
    txt = re.sub(r"\s+", " ", txt).strip()

    # robust markers (case-insensitive, with/without accent)
    up = txt.upper()
    start_m = re.search(r"\b(D[ÉE]DICACE|DEDICATION)\b", up)
    end_m   = re.search(r"\bTHE\s+END\b", up)

    if not start_m or not end_m or end_m.start() <= start_m.end():
        raise ValueError("Could not locate 'DEDICACE'…'THE END' region in the page.")

    core = txt[start_m.end():end_m.start()].strip()
    sents = length_filter(core)
    return filter_gutenberg(sents, min_tokens=min_tokens)

# ---- builders / samplers
def sample_matched_token_pairs(simple_sents, complex_sents, n=200, seed=SEED, key=bert_token_length):
    """ Greedy length-matching by BERT token length. """
    if len(simple_sents) < n or len(complex_sents) < n:
        raise ValueError(f"Need at least {n} items in both classes: have {len(simple_sents)} simple, {len(complex_sents)} complex")
    rng = random.Random(seed)

    # bucket by length
    sp_by_len, cp_by_len = defaultdict(list), defaultdict(list)
    for s in simple_sents:  sp_by_len[key(s)].append(s)
    for s in complex_sents: cp_by_len[key(s)].append(s)

    # build simple sample with length diversity
    lengths = sorted(sp_by_len.keys())
    sp_sample = []
    if len(lengths) >= n:
        for L in rng.sample(lengths, n):
            sp_sample.append(rng.choice(sp_by_len[L]))
    else:
        picks = {L: 0 for L in lengths}
        for L in lengths:
            sp_sample.append(rng.choice(sp_by_len[L])); picks[L] += 1
        remaining = n - len(lengths)
        caps = {L: len(sp_by_len[L]) for L in lengths}
        while remaining > 0:
            eligible = [L for L in lengths if picks[L] < caps[L]]
            if not eligible:
                break  # ran out; should not happen if initial size checks passed
            min_p = min(picks[L] for L in eligible)
            candidates = [L for L in eligible if picks[L] == min_p]
            chosen = rng.choice(candidates)
            sp_sample.append(rng.choice(sp_by_len[chosen])); picks[chosen] += 1
            remaining -= 1

    # match complex to the sampled simple lengths
    cp_available = {L: list(bucket) for L, bucket in cp_by_len.items()}
    cp_sample = []
    for s in sp_sample:
        L = key(s)
        bucket = cp_available.get(L)
        if bucket:
            cp_sample.append(bucket.pop(rng.randrange(len(bucket))))
        else:
            avail = sorted([L2 for L2, b in cp_available.items() if b])
            if not avail:
                raise ValueError("Ran out of complex sentences while matching lengths.")
            best = min(avail, key=lambda L2: (abs(L2 - L), -L2))
            cp_sample.append(cp_available[best].pop(rng.randrange(len(cp_available[best]))))

    return sp_sample, cp_sample

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

def _stats(s):
    w = _WORD.findall(s)
    if not w: return 0,0,0,0.0
    lens = [len(x) for x in w]
    S, C = sum(lens), len(lens)
    return S, C, max(lens), S/C

def _pctl(x, q=0.9):
    if not x: return 0
    x = sorted(x); k = int(q * (len(x) - 1))
    return x[k]

# core builder
def build_test_with_avg(simple_sentences, complex_sentences, N=100, cap=2000, max_fix=60, max_swaps=80, q=0.9):
    # annotate (L,C,M,A,s), keep only sentences with words
    S = [(*_stats(s), s) for s in simple_sentences if _stats(s)[1] > 0]
    C = [(*_stats(s), s) for s in complex_sentences if _stats(s)[1] > 0]
    if not S or not C: return [], []

    # Pick top-N simple by avg, bottom-N complex by avg
    S.sort(key=lambda t: (-t[3], -t[1], -t[0], -t[2]))  # A desc, tie-breaks
    C.sort(key=lambda t: ( t[3], -t[1], -t[0], -t[2]))  # A asc
    S_sel, C_sel = S[:N], C[:N]
    S_res, C_res = S[N:cap], C[N:cap]

    # aggregates
    SL, SC = sum(t[0] for t in S_sel), sum(t[1] for t in S_sel)
    CL, CC = sum(t[0] for t in C_sel), sum(t[1] for t in C_sel)
    avgS = lambda: SL/SC if SC else 0.0
    avgC = lambda: CL/CC if CC else 0.0

    # If avg constraint fails, do bounded swaps to fix (raise S avg / lower C avg)
    if avgS() < avgC():
        S_sel.sort(key=lambda t: t[3])         # low-avg first to replace
        S_res.sort(key=lambda t: -t[3])        # high-avg first to insert
        C_sel.sort(key=lambda t: -t[3])        # high-avg first to replace
        C_res.sort(key=lambda t: t[3])         # low-avg first to insert
        i = j = 0
        while avgS() < avgC() and max_fix > 0:
            changed = False
            if i < len(S_sel) and i < len(S_res):
                out, inn = S_sel[i], S_res[i]
                if inn[3] > out[3]:
                    SL += inn[0] - out[0]; SC += inn[1] - out[1]
                    S_sel[i], S_res[i] = inn, out
                    changed = True
                i += 1
            if avgS() < avgC() and j < len(C_sel) and j < len(C_res):
                out, inn = C_sel[j], C_res[j]
                if inn[3] < out[3]:
                    CL += inn[0] - out[0]; CC += inn[1] - out[1]
                    C_sel[j], C_res[j] = inn, out
                    changed = True
                j += 1
            if not changed: break
            max_fix -= 1

    # Match 90th percentile of per-sentence max-word-length
    p90 = lambda sel: _pctl([t[2] for t in sel], q)
    k = max(0, int(q * (N - 1)))  # index boundary for "top decile" block

    # pre-sort reserves for quick pulls
    C_res_hi = sorted(C_res, key=lambda t: (-t[2], t[3], -t[1], -t[0]))  # big max, avg-friendly
    S_res_hi = sorted(S_res, key=lambda t: (-t[2], -t[3], -t[1], -t[0]))
    S_res_lo = sorted(S_res, key=lambda t: ( t[2], -t[3], -t[1], -t[0]))  # small max
    C_res_lo = sorted(C_res, key=lambda t: ( t[2],  t[3], -t[1], -t[0]))

    def replace(sel, idx, inn, simple=True):
        nonlocal SL, SC, CL, CC
        out = sel[idx]
        if simple: SL += inn[0]-out[0]; SC += inn[1]-out[1]
        else:      CL += inn[0]-out[0]; CC += inn[1]-out[1]
        sel[idx] = inn
        return out

    swaps = 0
    while swaps < max_swaps:
        pS, pC = p90(S_sel), p90(C_sel)
        if pS == pC: break

        if pS > pC:
            # Raise complex p90 first: swap weakest max in C's top decile with a higher-max reserve
            C_top = sorted(range(k, N), key=lambda i: C_sel[i][2])[:6]
            improved = False
            for idx in C_top:
                for j, cand in enumerate(C_res_hi[:12]):
                    prev = pC
                    out = replace(C_sel, idx, cand, simple=False)
                    if p90(C_sel) >= prev and avgS() >= avgC():
                        C_res_hi[j] = out; swaps += 1; improved = True; break
                    C_sel[idx] = out; CL -= cand[0]-out[0]; CC -= cand[1]-out[1]  # revert
                if improved: break
            if not improved:
                # fallback: lower simple p90 slightly (swap biggest max from S top decile)
                S_top = sorted(range(k, N), key=lambda i: -S_sel[i][2])[:6]
                for idx in S_top:
                    for j, cand in enumerate(S_res_lo[:12]):
                        prev = pS
                        out = replace(S_sel, idx, cand, simple=True)
                        if p90(S_sel) <= prev and avgS() >= avgC():
                            S_res_lo[j] = out; swaps += 1; improved = True; break
                        S_sel[idx] = out; SL -= cand[0]-out[0]; SC -= cand[1]-out[1]
                if not improved: break
        else:
            # pC > pS: mirror
            S_top = sorted(range(k, N), key=lambda i: S_sel[i][2])[:6]
            improved = False
            for idx in S_top:
                for j, cand in enumerate(S_res_hi[:12]):
                    prev = pS
                    out = replace(S_sel, idx, cand, simple=True)
                    if p90(S_sel) >= prev and avgS() >= avgC():
                        S_res_hi[j] = out; swaps += 1; improved = True; break
                    S_sel[idx] = out; SL -= cand[0]-out[0]; SC -= cand[1]-out[1]
                if improved: break
            if not improved:
                C_top = sorted(range(k, N), key=lambda i: -C_sel[i][2])[:6]
                for idx in C_top:
                    for j, cand in enumerate(C_res_lo[:12]):
                        prev = pC
                        out = replace(C_sel, idx, cand, simple=False)
                        if p90(C_sel) <= prev and avgS() >= avgC():
                            C_res_lo[j] = out; swaps += 1; improved = True; break
                        C_sel[idx] = out; CL -= cand[0]-out[0]; CC -= cand[1]-out[1]
                if not improved: break

    Test_simple  = [t[4] for t in S_sel]
    Test_complex = [t[4] for t in C_sel]

    # Print statements
    SL2, SC2 = sum(_stats(s)[0] for s in Test_simple), sum(_stats(s)[1] for s in Test_simple)
    CL2, CC2 = sum(_stats(s)[0] for s in Test_complex), sum(_stats(s)[1] for s in Test_complex)
    aS, aC = (SL2/SC2 if SC2 else 0.0), (CL2/CC2 if CC2 else 0.0)
    pS, pC = _pctl([_stats(s)[2] for s in Test_simple], q), _pctl([_stats(s)[2] for s in Test_complex], q)
    print(f"[St_Wiki_WrdLenCtrl_Test_simple-avg] n={N} | avg_simple={aS:.4f} | avg_complex={aC:.4f}")
    print(f"[St_Wiki_WrdLenCtrl_Test_p90] p{int(q*100)}(maxlen): simple={pS} | complex={pC} | Δ={abs(pS-pC)} | swaps={swaps}")

    return Test_simple, Test_complex

# ---- dataset builder helpers ----
def get_dev_splits(n: int = 200):
    hegel_pairs = load_and_filter_gutenberg("https://www.gutenberg.org/ebooks/55108.txt.utf-8")
    kant_pairs  = load_and_filter_gutenberg("https://www.gutenberg.org/ebooks/4280.txt.utf-8")
    Dev_complex_pairs = hegel_pairs + kant_pairs
    Dev_complex = [s for _, s in Dev_complex_pairs]
    Dev_simple_prelim = load_and_filter_mctest("mc500")
    Dev_simple, Dev_complex = sample_matched_token_pairs(Dev_simple_prelim, Dev_complex, n=n)
    return Dev_simple, Dev_complex

def get_test_splits(n_len_ctrl: int = 100):
    Test_simple = load_and_filter_children_stories()
    Test_complex = load_and_filter_wiki()
    WLC_Test_simple, WLC_Test_complex = build_test_with_avg(Test_simple, Test_complex, N=n_len_ctrl, cap=2000)
    return Test_simple, Test_complex, WLC_Test_simple, WLC_Test_complex

def get_gau_split():
    GAU_pairs = load_and_filter_gutenberg_au()
    GAU_sents = [s for _, s in GAU_pairs]
    return GAU_sents

def get_all_datasets(n_dev: int = 200, n_len_ctrl: int = 100):
    Dev_simple, Dev_complex = get_dev_splits(n_dev)
    Test_simple, Test_complex, WLC_Test_simple, WLC_Test_complex = get_test_splits(n_len_ctrl)
    GAU_simple = get_gau_split()
    return {"Dev_simple": Dev_simple, "Dev_complex": Dev_complex,
        "Test_simple": Test_simple, "Test_complex": Test_complex,
        "WLC_Test_simple": WLC_Test_simple, "WLC_Test_complex": WLC_Test_complex,
        "GAU_simple": GAU_simple, "Abstractness_Test": Abstractness_Test}
