from .base import BaseMatchingAgent
from autogen import AssistantAgent, config_list_from_json
import os

class SemanticMatchAgent(BaseMatchingAgent):
    def __init__(self):
        config_list = [
            {
                "model": os.getenv("OPENAI_DEPLOYMENT_NAME"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_type": "azure",
                "api_version": os.getenv("OPENAI_API_VERSION")
            }
        ]
        self.llm = AssistantAgent(
            name="SemanticAgent",
            system_message="You are a renewable energy expert. Match product descriptions semantically.",
            llm_config={"config_list": config_list, "timeout": 120}
        )

    def match(self, extracted: str, etrm_df):
        results = []
        for _, row in etrm_df.iterrows():
            prompt = f"Compare extracted value '{extracted}' with ETRM description '{row['description']}'. Return a similarity score 0-100."
            response = self.llm.step(prompt)
            try:
                score = int("".join(filter(str.isdigit, response["content"])))
            except:
                score = 50
            results.append({"match": row["description"], "score": score, "method": "semantic"})
        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]
