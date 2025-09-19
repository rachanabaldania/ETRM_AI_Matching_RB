import os
from typing import Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
from .base_agent import BaseMatchingAgent
import pandas as pd

load_dotenv()

class AISemanticAgent(BaseMatchingAgent):
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
    
    def match(self, extracted_name: str, etrm_data: pd.DataFrame) -> Optional[str]:
        """Get AI assistance for matching when needed"""
        prompt = f"""
        You are a renewable energy sector expert and product matching specialist. 
        Help match the shipping product '{extracted_name}' to the most appropriate product from this list:
        
        {etrm_data['description'].tolist()}
        
        Consider:
        1. Exact name matches
        2. Partial matches
        3. abbreviations
        4. Industry terminology
        5. Similar products
        
        Return your best match and reasoning.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in product matching for the energy sector."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting AI assistance: {e}")
            return None