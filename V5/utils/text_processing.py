import re

# Common energy abbreviations dictionary
ABBREVIATIONS = {
    "RFG": "Reformulated Gasoline",
    "ULSD": "Ultra Low Sulfur Diesel",
    "AGO": "Automotive Gas Oil",
    "FO": "Fuel Oil",
    "LPG": "Liquefied Petroleum Gas",
    "NAP": "Naphtha",
    "JET": "Jet Fuel",
    "GN": "Gasoline",
    "OD": "Odorized",
    "NOD": "Non-Odorized"
}

def preprocess_text(text: str) -> str:
    """
    Basic preprocessing:
      - Lowercase
      - Remove special characters (except letters/numbers/spaces)
      - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep only alphanumeric + space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def expand_abbreviation(word: str) -> str:
    """
    Expand known abbreviations (e.g., ULSD -> Ultra Low Sulfur Diesel).
    Returns expanded string if found, else original.
    """
    return ABBREVIATIONS.get(word.upper(), word)

def normalize_description(desc: str) -> str:
    """
    Normalize full product description:
      - Preprocess
      - Expand abbreviations
    """
    if not desc:
        return ""
    desc = preprocess_text(desc)
    words = desc.split()
    expanded = [expand_abbreviation(w) for w in words]
    return " ".join(expanded)
