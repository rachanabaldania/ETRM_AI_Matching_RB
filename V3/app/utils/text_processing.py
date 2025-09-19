import re

def preprocess_text(text: str) -> str:
    """Preprocess text for matching by removing special chars and standardizing"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text