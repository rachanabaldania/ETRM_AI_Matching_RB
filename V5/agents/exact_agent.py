from .base import BaseMatchingAgent

class ExactMatchAgent(BaseMatchingAgent):
    def match(self, extracted: str, etrm_df):
        matches = etrm_df[etrm_df["description"].str.lower() == extracted.lower()]
        return [{"match": row["description"], "score": 100, "method": "exact"} for _, row in matches.iterrows()]
