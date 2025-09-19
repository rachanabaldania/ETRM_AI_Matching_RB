from difflib import SequenceMatcher
from .base_agent import BaseMatchingAgent
from app.utils.text_processing import preprocess_text

class SemanticMatchAgent(BaseMatchingAgent):
    def match(self, extracted: str, etrm_description: str) -> float:
        """Calculate similarity score between two strings"""
        return SequenceMatcher(
            None, 
            preprocess_text(extracted), 
            preprocess_text(etrm_description)
        ).ratio()