import pandas as pd
import numpy as np
import json
import re
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy.optimize import linear_sum_assignment
from typing import List, Dict

# 1. Load and Clean Data

def load_extracted_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['Product Name'] = df['Product Name'].astype(str).str.strip().str.upper()
    return df

def load_etrm_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df['description'] = df['description'].astype(str).str.strip().str.upper()
    return df

def preprocess_text(text):
    # Remove special chars, multiple spaces, etc.
    text = re.sub(r'[^A-Z0-9 ]', ' ', text.upper())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 2. Synonyms/Abbreviations
# (In production, maintain a domain synonym dictionary or use LLM for lookup)
ENERGY_SYNONYMS = {
    "LPG": ["LIQUEFIED PETROLEUM GAS", "LP GAS"],
    "NG": ["NATURAL GAS"],
    "PROPANE": ["UN1075", "LIQUEFIED PETROLEUM GASES ODORIZED (PROPANE)", "PROPANE GAS"],
    "ISO BUTANE": ["ISOBUTANE", "ISO BUTANE NON ODORIZED"],
    # Add as required
}

def expand_abbreviations(text: str) -> List[str]:
    expansions = [text]
    for abbr, fulls in ENERGY_SYNONYMS.items():
        if abbr in text:
            expansions.extend(fulls)
        for full in fulls:
            if full in text:
                expansions.append(abbr)
    return list(set(expansions))

# 3a. Rule-based Match
def rule_based_match(a, b):
    if a == b:
        return 1.0, "Exact"
    elif a in b or b in a:
        return 0.95, "Substring"
    elif any(k in b for k in a.split()):
        return 0.85, "Keyword"
    else:
        return 0.0, ""

def jaccard_sim(a, b):
    set1, set2 = set(a.split()), set(b.split())
    if not set1 or not set2:
        return 0.0
    inter = set1 & set2
    union = set1 | set2
    return len(inter) / len(union)

# 3b Fuzzy Token
def fuzzy_match(a, b):
    score = fuzz.token_sort_ratio(a, b)
    return score/100.0

# 3c Embedding Similarity (SBERT)
SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
def sbert_similarity(a, b):
    emb = SBERT_MODEL.encode([a, b])
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return score

# TF-IDF
def tfidf_match(a_list, b_list):
    tfidf = TfidfVectorizer().fit(a_list + b_list)
    a_vecs = tfidf.transform(a_list)
    b_vecs = tfidf.transform(b_list)
    return cosine_similarity(a_vecs, b_vecs)

# 4. Hungarian Matching (for best global assignment)
def hungarian_match(sim_matrix):
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)  # maximize
    return list(zip(row_ind, col_ind, sim_matrix[row_ind, col_ind]))

# 5. LLM Prompt for Abbreviation Expansion (for fallback)
def llm_expand_abbreviation(text: str) -> str:
    # In production: Use OpenAI/GPT here. For POC, return input or synonym expansion.
    expansions = expand_abbreviations(text)
    return expansions[-1] if len(expansions) > 1 else text

# 6. Label & Scoring
def label_score(score, method):
    if method == "Exact": return "Exact", 1.0
    if method == "Substring": return "Partial", score
    if score >= 0.95: return "Partial", score
    if score >= 0.80: return "Fuzzy", score
    if score >= 0.75: return "Semantic", score
    if score >= 0.50: return "Abbrev", score
    return "No Match", 0.0

# 7. Main Matching Pipeline
def match_products(extracted_df: pd.DataFrame, etrm_df: pd.DataFrame):
    results = []
    etrm_descs = etrm_df['description'].apply(preprocess_text).tolist()
    etrm_ids = etrm_df['id_number'].tolist()
    
    for idx, row in extracted_df.iterrows():
        prod = preprocess_text(str(row['Product Name']))
        # Try all expansions
        prod_expansions = expand_abbreviations(prod)
        # Precompute SBERT for product + all its expansions
        prod_embs = SBERT_MODEL.encode(prod_expansions)
        best_score = 0.0
        best_j = -1
        best_method = ""
        best_label = ""
        best_abbre = ""
        for j, desc in enumerate(etrm_descs):
            # 1. Rule-based
            for exp in prod_expansions:
                rb_score, rb_type = rule_based_match(exp, desc)
                if rb_score >= 1.0:
                    best_score, best_j, best_method, best_label = rb_score, j, "Rule-based", rb_type
                    break
                # 2. Jaccard
                jac = jaccard_sim(exp, desc)
                if jac > best_score:
                    best_score, best_j, best_method, best_label = jac, j, "Jaccard", "Partial"
                # 3. Fuzzy
                fuzz_score = fuzzy_match(exp, desc)
                if fuzz_score > best_score and fuzz_score >= 0.80:
                    best_score, best_j, best_method, best_label = fuzz_score, j, "Fuzzy", "Fuzzy"
                # 4. SBERT
            # SBERT for all prod_expansions to desc
            desc_emb = SBERT_MODEL.encode(desc)
            for exp_idx, exp_emb in enumerate(prod_embs):
                sbert_score = cosine_similarity([exp_emb], [desc_emb])[0][0]
                if sbert_score > best_score and sbert_score >= 0.75:
                    best_score, best_j, best_method, best_label = sbert_score, j, "SBERT", "Semantic"
        # If still not good match, try LLM for abbreviation
        if best_score < 0.75:
            llm_exp = llm_expand_abbreviation(prod)
            for j, desc in enumerate(etrm_descs):
                if llm_exp in desc:
                    best_score, best_j, best_method, best_label = 0.9, j, "LLM Abbrev", "Abbrev"
                    best_abbre = llm_exp
                    break
        # Assign
        if best_j >= 0:
            etrm_match = etrm_df.iloc[best_j]
            label, conf = label_score(best_score, best_label)
            results.append({
                "Extracted Product": row['Product Name'],
                "ETRM id_number": etrm_match['id_number'],
                "ETRM description": etrm_match['description'],
                "Score": round(float(best_score), 3),
                "Label": label,
                "Confidence": conf,
                "Match Method": best_method,
                "Abbreviation Used": best_abbre,
            })
        else:
            results.append({
                "Extracted Product": row['Product Name'],
                "ETRM id_number": "",
                "ETRM description": "",
                "Score": 0.0,
                "Label": "No Match",
                "Confidence": 0.0,
                "Match Method": "",
                "Abbreviation Used": "",
            })
    return results

# 8. Save Output
def save_results(results: List[Dict], out_csv: str, out_json: str):
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

# 9. Main Entrypoint
if __name__ == "__main__":
    # Provide your actual file paths
    extracted_csv = "C:\\Users\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\data\\Extracted_Data.xlsx"
    etrm_xlsx = "C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\data\\ETRM_Data.xlsx"
    out_csv = "C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\results\\matched_results.csv"
    out_json = "C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\results\\matched_results.json"
    print("Loading data...")
    extracted_df = load_extracted_data(extracted_csv)
    etrm_df = load_etrm_data(etrm_xlsx)
    print("Matching...")
    results = match_products(extracted_df, etrm_df)
    print("Saving results...")
    save_results(results, out_csv, out_json)
    print(f"Done. Results saved to {out_csv} and {out_json}")