from pydantic import BaseModel
from typing import Optional, List, Dict

class MatchResult(BaseModel):
    extracted_name: str
    matched_name: str
    matching_score: float
    reason: str
    etrm_code: str
    etrm_id: str
    alternatives: Optional[List[Dict]] = None
    ai_insight: Optional[str] = None

class MatchRequest(BaseModel):
    extracted_name: str