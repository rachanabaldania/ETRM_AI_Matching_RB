from typing import Iterable
from .base_agent import BaseMatchingAgent
from app.utils.text_processing import preprocess_text

class AbbreviationMatchAgent(BaseMatchingAgent):
    def match(self, extracted: str, etrm_description: str) -> bool:
        """Check if extracted text might be an abbreviation of ETRM description"""
        extracted_clean = preprocess_text(extracted)
        etrm_clean = preprocess_text(etrm_description)
        
        extracted_words = extracted_clean.split()
        etrm_words = etrm_clean.split()
        
        if not extracted_words:
            return False
            
        # Check initials
        extracted_initials = ''.join([word[0] for word in extracted_words if word])
        etrm_initials = ''.join([word[0] for word in etrm_words if word])
        
        if extracted_initials and extracted_initials in etrm_initials:
            return True
        
        # Check ordered words
        it = iter(etrm_words)
        return all(word in it for word in extracted_words)