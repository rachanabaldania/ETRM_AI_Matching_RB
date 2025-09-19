from .base import BaseMatchingAgent

class RuleAgent(BaseMatchingAgent):
    def match(self, extracted: str, etrm_df):
        # Rule: SME comments (if available)
        results = []
        for _, row in etrm_df.iterrows():
            if extracted.lower() in str(row["description"]).lower():
                results.append({
                    "match": row["description"],
                    "score": 95,
                    "method": "rule"
                })
        return results
