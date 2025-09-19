from .base import BaseMatchingAgent
import psycopg2
import os

class AbbreviationAgent(BaseMatchingAgent):
    def __init__(self):
        self.conn = psycopg2.connect(os.getenv("POSTGRES_URL"))

    def match(self, extracted: str, etrm_df):
        cur = self.conn.cursor()
        cur.execute("SELECT full_form FROM abbreviations WHERE abbr=%s", (extracted,))
        row = cur.fetchone()
        if row:
            full_form = row[0]
            matches = etrm_df[etrm_df["description"].str.contains(full_form, case=False, na=False)]
            return [{"match": m, "score": 90, "method": "abbreviation"} for m in matches["description"].tolist()]
        else:
            # Insert new abbreviation if discovered later
            return []
