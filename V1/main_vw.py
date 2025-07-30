"""product_matching_poc.py

End‑to‑end Proof‑of‑Concept for product name matching between extracted PDF
Shipping Documents and an ETRM (Energy Trading & Risk Management) master data set.

The script implements a layered matching architecture:
    1. Pre‑processing / normalisation of free‑text columns.
    2. Candidate generation via rule‑based filters.
    3. Three family of similarity scorers:
        3a. Rule‑based     – exact / substring / Jaccard (TF‑IDF cosine)
        3b. Fuzzy‑token    – RapidFuzz token_set_ratio
        3c. Embeddings     – SBERT cosine similarity
    4. Abbreviation logic – expands or builds abbreviations for comparison.
    5. Scoring & labelling – chooses the highest ranked hit and tags it as
       Exact / Abbrev / Partial / Semantic with a confidence value 0‑1.
    6. Persisting the enriched data set to CSV and JSON.

Dependencies
============
$ python -m pip install -r requirements.txt

pandas>=2.2.2
numpy>=1.26.4
rapidfuzz>=3.6.1
scikit-learn>=1.4.2
sentence-transformers>=2.7.0
python-Levenshtein (pulled in by rapidfuzz) 

Usage
=====
$ python product_matching_poc.py \
       --extracted_csv Extracted_Data.csv \
       --etrm_excel    ETRM_Data.xlsx \
       --out_csv       matched_output.csv \
       --out_json      matched_output.json

If the CLI args are omitted the script will look for the default file names in
its working directory.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process  # type: ignore
from sentence_transformers import SentenceTransformer, util  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
# Configuration
###############################################################################
DEFAULT_EXTRACTED_CSV = "C:\\Users\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\data\\Extracted_Data.xlsx"
DEFAULT_ETRM_XLSX = "C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\data\\ETRM_Data.xlsx"
DEFAULT_OUT_CSV =  "C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\results\\matched_results.csv"
DEFAULT_OUT_JSON =  "C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\results\\matched_results.json"

# Abbreviation expansion dictionary – extend as required for your domain
ABBREV_MAP: Dict[str, str] = {
    "LPG": "LIQUEFIED PETROLEUM GAS",
    "ISO": "ISOBUTANE",
    "NG": "NATURAL GAS",
    "RVP": "REID VAPOUR PRESSURE",
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384‑dim universal sentence model
EMBEDDING_THRESHOLD = 0.75
FUZZY_THRESHOLD = 80  # percentage
TFIDF_NGRAM_RANGE = (1, 3)

###############################################################################
# Utility functions
###############################################################################

punct_tbl = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Lower‑case, strip, remove punctuation + extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(punct_tbl)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_abbrev(text: str, mapping: Dict[str, str]) -> str:
    """Replace stand‑alone abbreviations with full forms."""
    tokens = text.split()
    expanded = [mapping.get(t.upper(), t) for t in tokens]
    return " ".join(expanded)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

###############################################################################
# Scorers
###############################################################################

class Scorer:
    """Base class for all similarity scorers."""

    label: str

    def score(self, query: str, candidate: str) -> float:  # noqa: D401 (simple)
        """Return similarity in [0, 1]."""
        raise NotImplementedError


class ExactScorer(Scorer):
    label = "Exact"

    def score(self, q: str, c: str) -> float:
        return 1.0 if normalise(q) == normalise(c) else 0.0


class SubstringScorer(Scorer):
    label = "Substring"

    def score(self, q: str, c: str) -> float:
        nq, nc = normalise(q), normalise(c)
        return 1.0 if nq in nc or nc in nq else 0.0


class FuzzyScorer(Scorer):
    label = "Fuzzy"

    def score(self, q: str, c: str) -> float:
        return fuzz.token_set_ratio(q, c) / 100.0


class TFIDFScorer(Scorer):
    label = "TFIDF"

    def __init__(self, corpus: List[str]):
        self.vectoriser = TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE)
        self.tfidf = self.vectoriser.fit_transform(corpus)

    def score(self, q: str, c: str) -> float:
        vec_q = self.vectoriser.transform([q])
        vec_c = self.vectoriser.transform([c])
        return cosine_similarity(vec_q, vec_c)[0, 0]


class EmbeddingScorer(Scorer):
    label = "Semantic"

    def __init__(self, sentences: List[str]):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embeddings = self.model.encode(sentences, convert_to_tensor=True)

    def score(self, q: str, c: str) -> float:
        emb_q = self.model.encode([q], convert_to_tensor=True)
        emb_c = self.model.encode([c], convert_to_tensor=True)
        return float(util.cos_sim(emb_q, emb_c)[0])

###############################################################################
# Matching engine
###############################################################################

def build_abbrev_set(text: str) -> set[str]:
    """Create a set of uppercase abbreviations derived from the text tokens."""
    tokens = [t for t in re.split(r"[^A-Za-z0-9]", text) if t]
    abbrevs = {t for t in tokens if 2 <= len(t) <= 6 and t.isupper()}
    # first‑letter abbreviation e.g. "LIQUEFIED PETROLEUM GAS" -> LPG
    words = [w for w in text.split() if w.isalpha() and len(w) > 2]
    if len(words) >= 2:
        abbrevs.add("".join(w[0].upper() for w in words))
    return abbrevs


def match_products(extracted: pd.DataFrame, etrm: pd.DataFrame) -> pd.DataFrame:
    # Pre‑compute scorer objects
    tfidf_scorer = TFIDFScorer(etrm["description"].fillna("").tolist())
    emb_scorer = EmbeddingScorer(etrm["description"].fillna("").tolist())

    # Build abbreviation index
    etrm["_abbrevs"] = etrm["description"].apply(build_abbrev_set)

    results: List[dict] = []

    for _, ext_row in extracted.iterrows():
        prod_name = ext_row["Product Name"]
        prod_norm = normalise(prod_name)
        prod_expanded = expand_abbrev(prod_name, ABBREV_MAP)

        best_score, best_label, best_idx = 0.0, "", None

        for idx, etrm_row in etrm.iterrows():
            desc = etrm_row["description"] or ""
            # Layer 1 – exact + substring
            s_exact = ExactScorer().score(prod_name, desc)
            if s_exact == 1.0:
                score, label = s_exact, "Exact"
            else:
                # Layer 2 – fuzzy
                s_fuzzy = FuzzyScorer().score(prod_name, desc)
                # Layer 3 – tfidf / jaccard
                s_tfidf = tfidf_scorer.score(prod_expanded, desc)
                # Layer 4 – embeddings
                s_sem = emb_scorer.score(prod_expanded, desc)

                # Use the best of the three secondary scores
                score_candidates: List[Tuple[float, str]] = [
                    (s_fuzzy, "Fuzzy"),
                    (s_tfidf, "Partial"),
                    (s_sem, "Semantic"),
                ]

                # Abbrev match – if any abbreviation appears in ETRM abbrev set
                abbrev_hit = bool(set(build_abbrev_set(prod_name)) & etrm_row["_abbrevs"])
                if abbrev_hit:
                    score_candidates.append((0.85, "Abbrev"))

                score, label = max(score_candidates, key=lambda x: x[0])

            if score > best_score:
                best_score, best_label, best_idx = score, label, idx

        if best_idx is not None:
            etrm_row = etrm.loc[best_idx]
            results.append(
                {
                    **ext_row.to_dict(),
                    "matched_id": etrm_row["id_number"],
                    "matched_description": etrm_row["description"],
                    "match_label": best_label,
                    "confidence": round(best_score, 3),
                }
            )
        else:
            # no match
            results.append(
                {
                    **ext_row.to_dict(),
                    "matched_id": None,
                    "matched_description": None,
                    "match_label": "NoMatch",
                    "confidence": 0.0,
                }
            )

    return pd.DataFrame(results)

###############################################################################
# Command‑line interface
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Product‑to‑ETRM matcher PoC")
    p.add_argument("--extracted_csv", default=DEFAULT_EXTRACTED_CSV)
    p.add_argument("--etrm_excel", default=DEFAULT_ETRM_XLSX)
    p.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    p.add_argument("--out_json", default=DEFAULT_OUT_JSON)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    extracted_df = pd.read_csv(args.extracted_csv)
    etrm_df = pd.read_excel(args.etrm_excel)

    matched_df = match_products(extracted_df, etrm_df)

    matched_df.to_csv(args.out_csv, index=False)
    matched_df.to_json(args.out_json, orient="records", indent=2)

    print(f"\nSaved {len(matched_df)} matched rows to {args.out_csv} and {args.out_json}.")


if __name__ == "__main__":
    main()
