# app/managers/matching_manager.py
import pandas as pd
import numpy as np
import json
import math
from typing import Dict, Optional, Any
from difflib import SequenceMatcher

class MatchingManager:
    def find_best_match(self, extracted_name: str, etrm_df: pd.DataFrame) -> Optional[Dict]:
        """Find the best match for extracted product name in ETRM data"""
        if etrm_df is None or etrm_df.empty:
            return None
            
        best_match = None
        best_score = 0
        alternatives = []
        
        for _, row in etrm_df.iterrows():
            etrm_description = str(row['description']) if pd.notna(row['description']) else ""
            
            # Calculate matching scores
            exact_score = 1.0 if self._exact_match(extracted_name, etrm_description) else 0
            partial_score = 0.8 if self._partial_match(extracted_name, etrm_description) else 0
            abbrev_score = 0.7 if self._abbreviation_match(extracted_name, etrm_description) else 0
            similarity_score = self._calculate_similarity(extracted_name, etrm_description)
            
            total_score = max(exact_score, partial_score, abbrev_score, similarity_score * 0.9)
            
            if total_score > best_score:
                best_score = total_score
                best_match = {
                    "extracted_name": extracted_name,
                    "matched_name": etrm_description,
                    "matching_score": float(round(best_score, 2)),
                    "reason": "",
                    "etrm_code": str(row['code']) if pd.notna(row['code']) else None,
                    "etrm_id": str(row['id_number']) if pd.notna(row['id_number']) else None
                }
                
            if total_score > 0.5:
                alternatives.append({
                    "matched_name": etrm_description,
                    "score": float(round(total_score, 2)),
                    "code": str(row['code']) if pd.notna(row['code']) else None,
                    "id": str(row['id_number']) if pd.notna(row['id_number']) else None
                })
        
        if best_match:
            if self._exact_match(extracted_name, best_match["matched_name"]):
                best_match["reason"] = "Exact name match"
            elif self._partial_match(extracted_name, best_match["matched_name"]):
                best_match["reason"] = "Partial name match"
            elif self._abbreviation_match(extracted_name, best_match["matched_name"]):
                best_match["reason"] = "Abbreviation match"
            else:
                best_match["reason"] = "Similarity match"
                
            best_match["alternatives"] = sorted(
                [alt for alt in alternatives if alt["matched_name"] != best_match["matched_name"]],
                key=lambda x: x["score"],
                reverse=True
            )[:5]
            
            return self._clean_nan_values(best_match)
        
        return None

    @staticmethod
    def _clean_nan_values(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively clean NaN, None, and other non-serializable values from dictionaries"""
        def clean(obj):
            if isinstance(obj, (str, int, bool)):
                return obj
            elif isinstance(obj, float):
                # Convert NaN/inf to None
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, (np.generic)):
                # Convert numpy types to native Python types
                return obj.item()
            elif isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean(item) for item in obj]
            elif obj is None:
                return None
            else:
                return str(obj)  # Fallback: convert to string
        
        cleaned = clean(data)
        return cleaned

    @staticmethod
    def _exact_match(extracted: str, etrm_description: str) -> bool:
        """Check for exact match"""
        return str(extracted).lower().strip() == str(etrm_description).lower().strip()

    @staticmethod
    def _partial_match(extracted: str, etrm_description: str) -> bool:
        """Check for partial match"""
        return str(extracted).lower().strip() in str(etrm_description).lower().strip()

    @staticmethod
    def _abbreviation_match(extracted: str, etrm_description: str) -> bool:
        """Check for abbreviation match"""
        extracted_clean = str(extracted).lower().strip()
        etrm_clean = str(etrm_description).lower().strip()
        
        extracted_words = extracted_clean.split()
        etrm_words = etrm_clean.split()
        
        if not extracted_words:
            return False
            
        extracted_initials = ''.join([word[0] for word in extracted_words if word])
        etrm_initials = ''.join([word[0] for word in etrm_words if word])
        
        if extracted_initials and extracted_initials in etrm_initials:
            return True
        
        it = iter(etrm_words)
        return all(word in it for word in extracted_words)

    @staticmethod
    def _calculate_similarity(extracted: str, etrm_description: str) -> float:
        """Calculate similarity score between two strings"""
        return SequenceMatcher(
            None, 
            str(extracted).lower().strip(), 
            str(etrm_description).lower().strip()
        ).ratio()