from .base_agent import BaseMatchingAgent
from app.utils.text_processing import preprocess_text

class PartialMatchAgent(BaseMatchingAgent):
    def match(self, extracted: str, etrm_description: str) -> bool:
        """Check if extracted text is a substring of ETRM description"""
        extracted_clean = preprocess_text(extracted)
        etrm_clean = preprocess_text(etrm_description)
        return extracted_clean in etrm_clean