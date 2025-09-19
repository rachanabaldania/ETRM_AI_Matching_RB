from .base import BaseMatchingAgent

class PartialMatchAgent(BaseMatchingAgent):
    def match(self, extracted: str, etrm_df):
        matches = etrm_df[etrm_df["description"].str.contains(extracted, case=False, na=False)]
        return [{"match": row["description"], "score": 70, "method": "partial"} for _, row in matches.iterrows()]
