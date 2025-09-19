from .base_agent import BaseMatchingAgent
from app.utils.text_processing import preprocess_text

class ExactMatchAgent(BaseMatchingAgent):
    def match(self, extracted: str, etrm_description: str) -> bool:
        """Check for exact match after preprocessing"""
        return preprocess_text(extracted) == preprocess_text(etrm_description)