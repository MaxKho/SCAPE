"""
Bundled datasets for SCAPE.

This module exposes dataset files placed in this package directory as importable
variables. Supported formats:

- *.txt       : one sentence per line → List[str], variable name from filename
                e.g. 'dev_simple.txt' → Dev_simple
- *.jsonl     : records with {"text", "label"(0/1) or "difficulty", "split"}
                → two List[str] variables '<stem>_simple' and '<stem>_complex'
                e.g. 'dev.jsonl' → Dev_simple, Dev_complex
- abstractness.csv : CSV with columns ('term','category','category_name')
                     → Dict[str, int] as 'Abstractness_Test'
                     and Dict[int, str] as 'Abstractness_Class_Names'

Utility:
- list_datasets() : tuple of public variable names
- as_dict()       : dict name -> object
"""

from __future__ import annotations
from importlib import resources as ir
from typing import Dict, List, Tuple
import json, csv

__all__: List[str] = []
_DATA_CACHE: Dict[str, object] = {}

# ---- helpers ----

_ACRONYMS = {"wlc", "tlc", "gau", "gpt2"}

def _pub_name_from_stem(stem: str) -> str:
    """
    Normalize a file stem into a public variable name:
      - capitalize tokens; known acronyms uppercased
      - keep trailing '_simple' / '_complex' lowercase
    Examples:
      'dev_simple' -> 'Dev_simple'
      'test_complex' -> 'Test_complex'
      'wlc_test_simple' -> 'WLC_Test_simple'
      'gau_simple' -> 'GAU_simple'
    """
    parts = stem.split("_")
    if not parts:
        return stem
    tail = ""
    if parts[-1] in ("simple", "complex"):
        tail = "_" + parts.pop()
    head = "_".join([p.upper() if p in _ACRONYMS else p.capitalize() for p in parts])
    return head + tail

def _read_txt(res) -> List[str]:
    with res.open("r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]
def _read_jsonl_paired(res) -> Tuple[List[str], List[str]]:
    simple, complex_ = [], []
    with res.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if text is None:
                continue
            label = rec.get("label")
            # accept numeric strings like "0"/"1"
            try:
                label_int = int(label) if label is not None else None
            except (TypeError, ValueError):
                label_int = None
            diff = (rec.get("difficulty") or "").lower()
            if label_int == 0 or diff == "simple":
                simple.append(text)
            elif label_int == 1 or diff == "complex":
                complex_.append(text)
    return simple, complex_

def _read_abstractness_csv(res):
    """
    Return:
      mapping: Dict[str, int]           # term/text -> category/class
      names:   Dict[int, str] | None    # category id -> category_name (if present)
    """
    with res.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = {c.lower(): c for c in (rdr.fieldnames or [])}

        # support both old ('text','class') and new ('term','category','category_name')
        term_key = cols.get("term") or cols.get("text")
        cat_key  = cols.get("category") or cols.get("class")
        name_key = cols.get("category_name") or cols.get("label")

        if not term_key or not cat_key:
            raise ValueError("abstractness.csv must have ('term','category') or ('text','class') columns")

        mapping: Dict[str, int] = {}
        names: Dict[int, str] = {}

        for row in rdr:
            term = row.get(term_key)
            if not term:
                continue
            try:
                cat = int(row.get(cat_key))
            except (TypeError, ValueError):
                continue
            mapping[term] = cat
            if name_key and row.get(name_key):
                names[cat] = row[name_key]

        return mapping, (names or None)

# ---- scan package resources ----

# NOTE: files() returns a Traversable; sorting for deterministic order is used.
_pkg = ir.files(__name__)

for entry in sorted(_pkg.iterdir(), key=lambda p: p.name.lower()):
    if not entry.is_file():
        continue
    name = entry.name
    stem, dot, ext = name.rpartition(".")
    ext = ext.lower()

    # special-case abstractness.csv
    if name.lower() == "abstractness.csv":
        mapping, names = _read_abstractness_csv(entry)
        globals()["Abstractness_Test"] = mapping
        __all__.append("Abstractness_Test")
        _DATA_CACHE["Abstractness_Test"] = mapping
        if names:
            globals()["Abstractness_Class_Names"] = names
            __all__.append("Abstractness_Class_Names")
            _DATA_CACHE["Abstractness_Class_Names"] = names
        continue


    if ext == "txt":
        pub = _pub_name_from_stem(stem)
        obj = _read_txt(entry)
        _DATA_CACHE[pub] = obj
        globals()[pub] = obj
        __all__.append(pub)
    elif ext == "jsonl":
        # derive paired names
        s_name = _pub_name_from_stem(stem + "_simple")
        c_name = _pub_name_from_stem(stem + "_complex")
        s, c = _read_jsonl_paired(entry)
        _DATA_CACHE[s_name] = s
        _DATA_CACHE[c_name] = c
        globals()[s_name] = s
        globals()[c_name] = c
        __all__.extend([s_name, c_name])

# ---- utilities ----

def list_datasets() -> Tuple[str, ...]:
    """Return the public dataset variable names available in this package."""
    return tuple(__all__)

def as_dict() -> Dict[str, object]:
    """Return a shallow copy of all datasets as a dict name -> object."""
    return dict(_DATA_CACHE)
