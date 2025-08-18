"""
core.py
-------

Core implementation of SCAPE-T (Semantic Complexity from Attention Patterns in 
Encoders - encoder-only Transformer variant).

This module provides the primary entry point for computing SCAPE-T scores. It 
defines the metric function itself (`scape_t`), handles model and tokeniser 
initialisation, and exposes default hyperparameters. 

Key features:
- A global `device` (picked once on import).
- Lazily-initialised singletons for the BERT `model` and `tokeniser`, exposed via
  `get_model()`, `get_tokeniser()`, and `get_model_tokeniser()`.
- `scape_t` function: computes semantic complexity scores for a given corpus or single text, with
  options for layer ablation or single-layer evaluation.
- Internal reliance on shared resources (abbreviations, delimiters, stopwords, 
  punctuation tables, default parameters) defined in `resources.py`.

Typical usage:
    from scape.scape_t.core import get_model, get_tokeniser, device, scape_t
    model = get_model()
    tok   = get_tokeniser()
    score = scape_t("Hello world")
"""

# ---- imports ----
from __future__ import annotations
import torch, re, threading
from typing import Optional, Tuple
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from .resources import ABBREVIATIONS, DELIMITERS, STOPWORDS, PUNCT, DEFAULT_PARAMS

# ---- global device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- lazy singletons (model & tokeniser) ----
__MODEL: Optional[BertModel] = None
__TOKENISER: Optional[BertTokenizer] = None
__INIT_LOCK = threading.Lock()

def _init_model_tokeniser_if_needed() -> None:
    """Initialise global BERT model & tokeniser once."""
    global __MODEL, __TOKENISER
    if __MODEL is not None and __TOKENISER is not None:
        return
    with __INIT_LOCK:
        if __MODEL is None or __TOKENISER is None:
            tok = BertTokenizer.from_pretrained("bert-base-uncased")
            mdl = BertModel.from_pretrained(
                "bert-base-uncased",
                output_attentions=True
            ).to(device).eval()
            __TOKENISER = tok
            __MODEL = mdl

def get_model() -> BertModel:
    """Return the shared BERT model (lazy-loaded)."""
    _init_model_tokeniser_if_needed()
    # mypy: __MODEL will be set now
    return __MODEL  # type: ignore[return-value]

def get_tokeniser() -> BertTokenizer:
    """Return the shared BERT tokeniser (lazy-loaded)."""
    _init_model_tokeniser_if_needed()
    return __TOKENISER  # type: ignore[return-value]

def get_model_tokeniser() -> Tuple[BertModel, BertTokenizer]:
    """Return the shared (model, tokeniser) pair (lazy-loaded)."""
    _init_model_tokeniser_if_needed()
    return __MODEL, __TOKENISER  # type: ignore[return-value]

def reset_model_tokeniser() -> None:
    """Dispose and reset the cached model/tokeniser (e.g., to switch device)."""
    global __MODEL, __TOKENISER
    with __INIT_LOCK:
        __MODEL = None
        __TOKENISER = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ---- text preprocessing ----
def filter_text(text):
    """Remove uninformative text specific to text from Project Gutenberg"""
    original_text = text
    if text and text[0] == "'" and (text.count("'") - text.count("'s") - text.count("'d")) % 2:
        text = text[1:].strip()
        print(f"[filter_text] Removed leading unmatched single quote from: '{original_text}'")
    if text and text[0] == '"' and text.count('"') % 2:
        text = text[1:].strip()
        print(f"[filter_text] Removed leading unmatched double quote from: '{original_text}'")
    if re.match(r"^\s*\d+", text):
        print(f"[filter_text] Removed leading number from: '{text}'")
        text = re.sub(r"^\s*\d+", "", text)
    while text and text[0] in {',', ':', ';', '.', '-'}:
        print(f"[filter_text] Removed leading punctuation '{text[0]}' from: '{text}'")
        text = text[1:].strip()
    for pattern, desc in [
        (r"^([ivxIVX]+|page|Page)\s+(page\s+)?\d+", "Roman numeral or page number"),
        (r"^line\s+\d+\s+from\s+(bottom|top)", "line number from top/bottom"),
        (r",\s*([ivxIVX]+|page)\s+\d+\s*\.", "trailing Roman numeral/page ref"),
        (r"^\d+\s+", "leading number"),
        (r"^'\s*,", "odd punctuation at start"),
        (r"^[ivxIVX]{2,}\s+", "Roman numeral only"),
    ]:
        if re.search(pattern, text):
            print(f"[filter_text] Matched and removed: {desc} in '{text}'")
            text = re.sub(pattern, "", text).strip()
    return text.strip()

def preprocess_sentence(text: str) -> str:
    """Remove text that the metric cannot handle"""
    original_text = text

    if re.search(r"(?:^|\s)'[^']*'(?:\s|$)", text):
        print(f"[preprocess_sentence] Discarded due to unmatched or isolated apostrophe usage: '{text}'")
        print("→ Please rewrite this sentence without quotation or direct speech.")
        return ""

    throw_away_sentence_markers = [
        '•', '”', '"', '§', " ' ", " i "
    ]
    for marker in throw_away_sentence_markers:
        if marker in text:
            print(f"[preprocess_sentence] Discarded due to marker '{marker}': '{text}'")
            print("→ Please rewrite this sentence without quotation, direct speech, formatting, or special characters.")
            return ""

    if re.match(r"^\s*\d+\s*:\s*", text):
        print(f"[preprocess_sentence] Discarded due to timestamp-like pattern: '{text}'")
        print("→ Please rewrite this without timestamps.")
        return ""

    for pattern, desc in [
        (r"[_\])]", "trailing underscore/bracket"),
        (r"^\s*[\[(]", "leading bracket"),
        (r"\s*[\[(]", "inline bracket"),
        (r"^\s*[,:;]", "leading punctuation"),
        (r"(\w)--", "missing space before em-dash"),
        (r"--(\w)", "missing space after em-dash"),
        (r"([,:;.!?]\s*)([,:;.!?]\s*)+", "redundant punctuation"),
        (r"-+-", "redundant dashes"),
        (r"^\s*Page(s?)\s+\d+", "Page number at start"),
        (r"^\s*'\s*'", "empty quote pair"),
        (r"^\s*\d+", "leading number"),
        (r"(\*\s*)+", "asterisk bullet"),
    ]:
        if re.search(pattern, text):
            print(f"[preprocess_sentence] Cleaned: {desc} in '{text}'")
        text = re.sub(pattern, "", text)

    # Filtering loop
    changed = True
    while changed:
        new_text = filter_text(text)
        changed = (text != new_text)
        if changed:
            print(f"[preprocess_sentence] Cleaned via filter_text: '{text}' → '{new_text}'")
        text = new_text

    text = re.sub(r"\s+\s", " ", text)
    text = text.strip()

    if text:
        last_char = text[-1]
        if last_char not in DELIMITERS:
            text += " ."
        text = re.sub(r"(?:[,:;.!?]\s*)+([,:;.!?]\s*)", r'\1', text)

    if re.match(r'^[a-z]', text):
        print(f"[preprocess_sentence] Discarded poorly formed sentence: '{text}'")
        print("→ Please rewrite this sentence to begin with a proper capital letter.")
        return ""

    return text

def preprocess_entire_text(text: str) -> str:
    """Replace misleading punctuation."""
    text = re.sub(r"\.\.\.(?=(.*?))", lambda m: (
        ". " if re.search(r"[A-Z]", m.group(1).lstrip()) else
        " " if re.search(r"[a-z]", m.group(1).lstrip()) else "..."), text)
    for reg_ex, repl in ABBREVIATIONS:
        text = re.sub(reg_ex, repl, text)
    text = re.sub(r"([.,;:?!])[.,;:?!]+", r"\1", text)
    return text

def split_text(text: str) -> list[str]:
    """Segment a string into words and punctuation."""
    text = preprocess_entire_text(text)
    return re.findall(r"[^\s.,;:?!]+|[.,;:?!]", text)

def group_into_sentences(words: list[str]) -> list[list[str]]:
    """Group a flat segment list into sentences"""
    res = []
    curr_sen = []
    for word in words:
        if word in DELIMITERS:
            if curr_sen:
                curr_sen.append(word)
                res.append(curr_sen)
                curr_sen = []
        else:
            curr_sen.append(word)
    if curr_sen:
        res.append(curr_sen)
    return res

def group_text(segments):
    """Convert a list of segments into sentence strings."""
    groups = group_into_sentences(segments)
    sentences = [' '.join(g) for g in groups]
    sentences = [preprocess_sentence(s) for s in sentences if s]
    return sentences

# ---- SCAPE-T core ----
def mask_attention(attn):
    """Zero out self-attention and attention to CLS token."""
    H, N, _ = attn.shape
    attn = attn.clone()
    idx = torch.arange(N, device=attn.device)
    attn[:, idx, idx] = 0.
    attn[:, :, 0]     = 0.
    return attn

def compute_head_redundancy(attn: torch.Tensor, prev_layers: list[torch.Tensor], p) -> torch.Tensor:
    """Compute redundancy-adjusted weights for attention heads in a layer."""
    H = attn.shape[0]
    vec = attn.view(H, -1)
    vec /= (vec.norm(dim=1, keepdim=True) + 1e-12)

    # Similarity to earlier heads in the same layer (j < i)
    sim_same = vec @ vec.T
    mask     = torch.tril(torch.ones_like(sim_same, dtype=torch.bool), diagonal=-1)
    sim_same = torch.where(mask, sim_same, torch.full_like(sim_same, -1.0))
    max_sim, _  = sim_same.max(dim=1)

    if prev_layers:
        prev_vec = torch.cat([pl.view(pl.shape[0], -1) for pl in prev_layers], dim=0)
        prev_vec /= (prev_vec.norm(dim=1, keepdim=True) + 1e-12)
        prev_max, _ = (vec @ prev_vec.T).max(dim=1)
        max_sim = torch.max(max_sim, prev_max)

    # Similarity to heads from previous layers.
    w = torch.where(max_sim < p['tau'], torch.ones_like(max_sim), 1.0 - max_sim)
    w[0] = 1.0
    return w

def compute_self_focus(attn):
    """Returns per-head self-focus values."""
    H, N, _       = attn.shape
    diag          = attn[:, torch.arange(N, device=attn.device), torch.arange(N, device=attn.device)]
    self_mass     = diag[:, 1:].mean(1)             # exclude CLS token
    off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=attn.device)
    nonself_mass  = attn[:, off_diag_mask].view(H, -1).mean(1)
    self_foci     = self_mass / (nonself_mass + 1e-12)
    return self_foci

def compute_cls_focus(attn):
    """Returns per-head CLS-focus values."""
    H, N, _ = attn.shape
    cls_foci = attn[:, :, 0].mean(dim=1) / attn[:, :, 1:].mean(dim=(1,2))
    return cls_foci

def combine(attn, prev_layers, p):
    """Combine redundancy, CLS focus, and self-focus penalty into head weights."""
    seq_len      = attn.shape[1]
    masked       = mask_attention(attn)
    redundancy   = compute_head_redundancy(masked, prev_layers, p)
    cls_foci = compute_cls_focus(attn)
    head_weights = (seq_len - 1) ** (-p['beta'] * p['zeta']) * redundancy ** p['alpha'] * (1 + cls_foci) ** p['beta']
    return head_weights.mean()

def layer_weights(n_layers, mu, sigma, eta, ablation=None, only_layer=None):
    """Generate normalised Gaussian weights centered at layer mu."""
    x = torch.arange(n_layers, device=device)
    g = torch.exp(-((x - mu)**2) / (2 * sigma**2))
    g /= g.sum()
    u = torch.ones(n_layers, device=device) / n_layers
    w = (1 - eta) * u + eta * g

    if only_layer is not None:
        w = w.clone()
        w[:] = 0.0
        w[int(only_layer) % n_layers] = 1.0
        return w

    if ablation is not None:
        w[int(ablation) % n_layers] = 0.0
        w = w / w.sum().clamp_min(1e-12)
    return w

def aggregate(attentions, p, layer_abl=None, only_layer=None):
    """Aggregate layer scores using layer weights."""
    scores, prev = [], []
    for attn in attentions:
        layer = attn[0]
        scores.append(combine(layer, prev, p))
        prev.append(mask_attention(layer))
    L = len(attentions)
    w = layer_weights(L, p['mu'], p['sigma'], p['eta'], layer_abl, only_layer).to(device)
    return (w * torch.stack(scores)).sum()

@torch.no_grad()
def scape_t(text_data, p=DEFAULT_PARAMS, layer_abl=None, only_layer=None):
    """Compute semantic complexity of a given text or corpus."""
    model = get_model()
    tokeniser = get_tokeniser()
    scores = {}
    corpus = text_data if isinstance(text_data, (dict, list, tuple, set)) else [text_data]
    for text in corpus:
        sentences = group_text(split_text(text))
        sentences = [' '.join([w for w in c.translate(PUNCT).split() if w.lower() not in STOPWORDS]) for c in sentences if c]
        sentence_scores = []
        for sentence in sentences:
            enc     = tokeniser([sentence], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**enc)
            atts    = outputs.attentions
            num_L   = len(atts)

            w = layer_weights(num_L, p['mu'], p['sigma'], p['eta'], layer_abl, only_layer).to(device)
            self_focus = (torch.stack([compute_self_focus(l[0]) for l in atts]).mean(1) * w).sum()
            base   = aggregate(atts, p, layer_abl, only_layer)
            input_ids = enc["input_ids"][0]
            score  = 100 * (base ** p['gamma']) * (1 + p['kappa'] / (self_focus + 1e-12)) ** p['delta']
            sentence_scores.append(score)

        if sentences: scores[text] = torch.stack(sentence_scores).mean().item() if sentence_scores else 0.

    return scores if isinstance(text_data, (dict, list, tuple, set)) else scores[text_data]
