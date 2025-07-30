import pandas as pd
import re
from fuzzywuzzy import fuzz, process
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

class ProductMatcher:
    def __init__(self):
        # Initialize the Sentence-BERT model for semantic matching
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.etrm_products = None
        self.etrm_embeddings = None
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text for matching"""
        if not isinstance(text, str):
            return ""
        text = text.upper().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
        text = re.sub(r'\b(?:INT|US|TRADING|CA|USA)\b', '', text)  # Remove common stopwords
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        return text.strip()
    
    def load_etrm_data(self, etrm_file: str) -> None:
        """Load and preprocess ETRM product data"""
        # Read ETRM products sheet (adjust sheet name as needed)
        etrm_df = pd.read_excel(etrm_file, sheet_name='Products')
        
        # Create clean versions for matching
        etrm_df['clean_description'] = etrm_df['description'].apply(self.preprocess_text)
        etrm_df['clean_name'] = etrm_df['name'].apply(self.preprocess_text)
        
        # Generate embeddings for semantic matching
        descriptions = etrm_df['clean_description'].tolist()
        self.etrm_embeddings = self.model.encode(descriptions)
        
        self.etrm_products = etrm_df
    
    def match_product(self, product_name: str) -> Dict:
        """Match a product name against ETRM database"""
        clean_query = self.preprocess_text(product_name)
        
        # Stage 1: Exact matching
        exact_matches = self.etrm_products[
            (self.etrm_products['clean_description'] == clean_query) | 
            (self.etrm_products['clean_name'] == clean_query)
        ]
        
        if not exact_matches.empty:
            best_match = exact_matches.iloc[0]
            return {
                'input_product': product_name,
                'matched_name': best_match['name'],
                'matched_description': best_match['description'],
                'match_type': 'exact',
                'confidence': 1.0
            }
        
        # Stage 2: Fuzzy matching
        choices = self.etrm_products['clean_description'].tolist()
        fuzzy_match = process.extractOne(clean_query, choices, scorer=fuzz.token_set_ratio, score_cutoff=80)
        
        if fuzzy_match:
            matched_desc, score = fuzzy_match
            best_fuzzy = self.etrm_products[self.etrm_products['clean_description'] == matched_desc].iloc[0]
            return {
                'input_product': product_name,
                'matched_name': best_fuzzy['name'],
                'matched_description': best_fuzzy['description'],
                'match_type': 'fuzzy',
                'confidence': score/100
            }
        
        # Stage 3: Semantic matching
        query_embedding = self.model.encode([clean_query])
        cos_sim = cosine_similarity(query_embedding, self.etrm_embeddings)[0]
        best_idx = np.argmax(cos_sim)
        best_score = cos_sim[best_idx]
        
        if best_score > 0.7:  # Semantic similarity threshold
            best_match = self.etrm_products.iloc[best_idx]
            return {
                'input_product': product_name,
                'matched_name': best_match['name'],
                'matched_description': best_match['description'],
                'match_type': 'semantic',
                'confidence': best_score
            }
        
        # No match found
        return {
            'input_product': product_name,
            'matched_name': None,
            'matched_description': None,
            'match_type': 'no_match',
            'confidence': 0.0
        }
    
    def process_extracted_data(self, extracted_file: str) -> pd.DataFrame:
        """Process the extracted data file and match products"""
        # Read extracted data (adjust columns as needed)
        extracted_df = pd.read_excel(extracted_file)
        
        # Get unique product names to match
        products_to_match = extracted_df['Product Name'].unique()
        
        # Match each product
        results = []
        for product in products_to_match:
            match_result = self.match_product(product)
            results.append(match_result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original data
        final_df = extracted_df.merge(
            results_df, 
            left_on='Product Name', 
            right_on='input_product',
            how='left'
        )
        
        return final_df

# Example usage
if __name__ == "__main__":
    matcher = ProductMatcher()
    
    # Load ETRM product data
    matcher.load_etrm_data("C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\data\\ETRM_Data.xlsx")  # Replace with your ETRM file
    
    # Process extracted data
    matched_results = matcher.process_extracted_data("C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\data\\Extracted_Data.xlsx")
    
    # Save results
    matched_results.to_excel("C:\\Users\\RachanaBaldania\\OneDrive - RandomTrees\\Rachana_Code\\ETRM_AI_Matching_RB\\data\\Matched_Results.xlsx", index=False)
    
    print("Matching completed. Results saved to Matched_Results.xlsx")