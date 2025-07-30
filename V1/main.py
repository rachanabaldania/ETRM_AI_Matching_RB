import pandas as pd
import numpy as np
import json
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import re
import warnings
import os
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
# Configuration - moved to top for better visibility
CONFIG = {
    "sbert_model": "all-mpnet-base-v2",
    "fuzzy_threshold": 80,
    "embedding_threshold": 0.75,
    "openai_model": "gpt-4",
    "output_formats": ["csv", "json"],
    "abbreviation_prompt": """You are an expert in the energy sector..."""  # (keep your existing prompt)
}


class ProductMatcher:
    def __init__(self, etrm_data_path: str):
        """
        Initialize the ProductMatcher with ETRM data.
        
        Args:
            etrm_data_path: Path to the ETRM data file
        """
        self.etrm_df = self._load_etrm_data(etrm_data_path)
        self.sbert_model = SentenceTransformer(CONFIG["sbert_model"])
        self.etrm_embeddings = None
        self.abbreviation_map = {}
        
    def _load_etrm_data(self, path: str) -> pd.DataFrame:
        """Load and preprocess ETRM data."""
        df = pd.read_excel(path)
        
        # Basic preprocessing
        df['description'] = df['description'].str.strip().str.lower()
        df['name'] = df['name'].str.strip().str.lower()
        df['code'] = df['code'].astype(str).str.strip()
        
        # Remove inactive products if needed
        df = df[df['active'] == 1]
        
        return df
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for matching."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        text = re.sub(r'[^\w\s-]', '', text)  # Remove special chars except hyphens
        text = re.sub(r'\s+', ' ', text)      # Collapse multiple spaces
        return text
    
    def _generate_embeddings(self):
        """Generate SBERT embeddings for ETRM descriptions."""
        if self.etrm_embeddings is None:
            descriptions = self.etrm_df['description'].tolist()
            self.etrm_embeddings = self.sbert_model.encode(descriptions)
    
    def _rule_based_match(self, query: str) -> Optional[Dict]:
        """
        Perform rule-based matching (exact, substring, keyword).
        
        Args:
            query: Product name to match
            
        Returns:
            Dictionary with match info or None if no match
        """
        query = self._preprocess_text(query)
        
        # Check for exact match
        exact_matches = self.etrm_df[self.etrm_df['description'] == query]
        if not exact_matches.empty:
            best_match = exact_matches.iloc[0]
            return {
                'method': 'exact',
                'match': best_match['description'],
                'id': best_match['id_number'],
                'code': best_match['code'],
                'confidence': 1.0,
                'type': 'Exact'
            }
        
        # Check for substring match
        substring_matches = self.etrm_df[self.etrm_df['description'].str.contains(query, regex=False)]
        if not substring_matches.empty:
            best_match = substring_matches.iloc[0]
            return {
                'method': 'substring',
                'match': best_match['description'],
                'id': best_match['id_number'],
                'code': best_match['code'],
                'confidence': 0.9,
                'type': 'Partial'
            }
        
        # Check for keyword match using Jaccard similarity
        query_words = set(query.split())
        max_sim = 0
        best_match = None
        
        for _, row in self.etrm_df.iterrows():
            desc_words = set(row['description'].split())
            intersection = query_words.intersection(desc_words)
            union = query_words.union(desc_words)
            jaccard = len(intersection) / len(union) if union else 0
            
            if jaccard > max_sim:
                max_sim = jaccard
                best_match = row
        
        if max_sim > 0.5:  # Threshold for keyword match
            return {
                'method': 'keyword_jaccard',
                'match': best_match['description'],
                'id': best_match['id_number'],
                'code': best_match['code'],
                'confidence': max_sim,
                'type': 'Partial'
            }
        
        return None
    
    def _fuzzy_token_match(self, query: str) -> Optional[Dict]:
        """
        Perform fuzzy token matching using RapidFuzz.
        
        Args:
            query: Product name to match
            
        Returns:
            Dictionary with match info or None if no match
        """
        query = self._preprocess_text(query)
        descriptions = self.etrm_df['description'].tolist()
        
        # Get best match using token set ratio
        best_match, score, idx = process.extractOne(
            query, 
            descriptions, 
            scorer=fuzz.token_set_ratio,
            score_cutoff=CONFIG["fuzzy_threshold"]
        )
        
        if score >= CONFIG["fuzzy_threshold"]:
            matched_row = self.etrm_df[self.etrm_df['description'] == best_match].iloc[0]
            return {
                'method': 'fuzzy_token',
                'match': best_match,
                'id': matched_row['id_number'],
                'code': matched_row['code'],
                'confidence': score / 100,
                'type': 'Partial'
            }
        
        return None
    
    def _embedding_similarity_match(self, query: str) -> Optional[Dict]:
        """
        Perform embedding-based similarity matching using SBERT.
        
        Args:
            query: Product name to match
            
        Returns:
            Dictionary with match info or None if no match
        """
        self._generate_embeddings()
        query = self._preprocess_text(query)
        
        # Encode query
        query_embedding = self.sbert_model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.etrm_embeddings)[0]
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        
        if max_sim >= CONFIG["embedding_threshold"]:
            best_match = self.etrm_df.iloc[max_idx]
            return {
                'method': 'sbert_embedding',
                'match': best_match['description'],
                'id': best_match['id_number'],
                'code': best_match['code'],
                'confidence': max_sim,
                'type': 'Semantic'
            }
        
        return None
    
    def _hungarian_matching(self, queries: List[str]) -> List[Dict]:
        """
        Perform optimal matching between multiple queries and ETRM products using Hungarian algorithm.
        
        Args:
            queries: List of product names to match
            
        Returns:
            List of match dictionaries
        """
        self._generate_embeddings()
        processed_queries = [self._preprocess_text(q) for q in queries]
        
        # Encode all queries
        query_embeddings = self.sbert_model.encode(processed_queries)
        
        # Calculate similarity matrix
        sim_matrix = cosine_similarity(query_embeddings, self.etrm_embeddings)
        
        # Apply Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        
        matches = []
        for r, c in zip(row_ind, col_ind):
            query = queries[r]
            similarity = sim_matrix[r, c]
            
            if similarity >= CONFIG["embedding_threshold"]:
                matched_row = self.etrm_df.iloc[c]
                matches.append({
                    'query': query,
                    'method': 'hungarian',
                    'match': matched_row['description'],
                    'id': matched_row['id_number'],
                    'code': matched_row['code'],
                    'confidence': similarity,
                    'type': 'Semantic'
                })
            else:
                matches.append({
                    'query': query,
                    'method': 'hungarian',
                    'match': None,
                    'id': None,
                    'code': None,
                    'confidence': similarity,
                    'type': 'No Match'
                })
        
        return matches
    
    def _llm_validate_match(self, query: str, candidate: str) -> Tuple[bool, float]:
        """
        Use LLM to validate a potential match.
        
        Args:
            query: Original product name
            candidate: Potential match from ETRM
            
        Returns:
            Tuple of (is_valid, confidence)
        """
        prompt = f"""You are an expert in petroleum products and chemicals. 
Determine if the following two product names refer to the same product:

1. Original: {query}
2. Candidate: {candidate}

Respond with JSON containing:
- "is_match": boolean (true if they refer to the same product)
- "confidence": float (0-1 how confident you are)
- "reason": brief explanation of your decision"""
        
        try:
            response = openai.ChatCompletion.create(
                model=CONFIG["openai_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('is_match', False), result.get('confidence', 0.5)
        except:
            return False, 0.5
    
    def _resolve_abbreviations(self, product_names: List[str]) -> Dict[str, str]:
        """
        Identify and resolve abbreviations in product names using LLM.
        
        Args:
            product_names: List of product names to analyze
            
        Returns:
            Dictionary mapping abbreviations to full forms
        """
        prompt = CONFIG["abbreviation_prompt"].format(product_names="\n".join(product_names))
        
        try:
            response = openai.ChatCompletion.create(
                model=CONFIG["openai_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            abbreviations = json.loads(response.choices[0].message.content)
            self.abbreviation_map = {item['abbreviation'].lower(): item['full_form'].lower() for item in abbreviations}
            return self.abbreviation_map
        except:
            return {}
    
    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand known abbreviations in text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with abbreviations expanded
        """
        if not self.abbreviation_map:
            return text
            
        words = text.lower().split()
        expanded_words = []
        
        for word in words:
            expanded = self.abbreviation_map.get(word, word)
            expanded_words.append(expanded)
            
        return ' '.join(expanded_words)
    
    def match_product(self, product_name: str, use_llm: bool = False) -> Dict:
        """
        Match a single product name to ETRM data using multiple techniques.
        
        Args:
            product_name: Product name to match
            use_llm: Whether to use LLM for final validation
            
        Returns:
            Dictionary with match information
        """
        # Try different matching techniques in order of precision
        match = self._rule_based_match(product_name)
        
        if not match:
            match = self._fuzzy_token_match(product_name)
        
        if not match:
            match = self._embedding_similarity_match(product_name)
        
        # If we have a match, optionally validate with LLM
        if match and use_llm:
            is_valid, llm_confidence = self._llm_validate_match(product_name, match['match'])
            if is_valid:
                match['confidence'] = (match['confidence'] + llm_confidence) / 2
                match['llm_validated'] = True
            else:
                match = None
        
        # Prepare result
        if match:
            result = {
                'input_product': product_name,
                'matched_product': match['match'],
                'match_type': match['type'],
                'confidence': match['confidence'],
                'method': match['method'],
                'etrm_id': match['id'],
                'etrm_code': match['code'],
                'llm_validated': match.get('llm_validated', False)
            }
        else:
            result = {
                'input_product': product_name,
                'matched_product': None,
                'match_type': 'No Match',
                'confidence': 0,
                'method': None,
                'etrm_id': None,
                'etrm_code': None,
                'llm_validated': False
            }
        
        return result
    
    def match_products(self, product_names: List[str], batch_mode: bool = True, use_llm: bool = False) -> List[Dict]:
        """
        Match multiple product names to ETRM data.
        
        Args:
            product_names: List of product names to match
            batch_mode: Whether to use Hungarian algorithm for batch processing
            use_llm: Whether to use LLM for final validation
            
        Returns:
            List of match dictionaries
        """
        # First resolve abbreviations
        self._resolve_abbreviations(product_names)
        
        if batch_mode and len(product_names) > 1:
            # Use Hungarian algorithm for optimal batch matching
            matches = self._hungarian_matching(product_names)
            
            if use_llm:
                for match in matches:
                    if match['match']:
                        is_valid, llm_confidence = self._llm_validate_match(
                            match['query'], 
                            match['match']
                        )
                        if is_valid:
                            match['confidence'] = (match['confidence'] + llm_confidence) / 2
                            match['llm_validated'] = True
                        else:
                            match.update({
                                'matched_product': None,
                                'match_type': 'No Match',
                                'confidence': 0,
                                'etrm_id': None,
                                'etrm_code': None,
                                'llm_validated': False
                            })
            
            # Convert to consistent format
            results = []
            for match in matches:
                results.append({
                    'input_product': match['query'],
                    'matched_product': match['match'],
                    'match_type': match['type'],
                    'confidence': match['confidence'],
                    'method': match['method'],
                    'etrm_id': match['id'],
                    'etrm_code': match['code'],
                    'llm_validated': match.get('llm_validated', False)
                })
            
            return results
        else:
            # Process individually
            return [self.match_product(name, use_llm) for name in product_names]
    
    def save_results(self, results: List[Dict], base_filename: str):
        """
        Save matching results to files.
        
        Args:
            results: List of match dictionaries
            base_filename: Base filename for output (without extension)
        """
        df = pd.DataFrame(results)
        
        if "csv" in CONFIG["output_formats"]:
            csv_path = f"{base_filename}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
        
        if "json" in CONFIG["output_formats"]:
            json_path = f"{base_filename}.json"
            df.to_json(json_path, orient='records', indent=2)
            print(f"Results saved to {json_path}")
def process_extracted_data(extracted_data_path: str, etrm_data_path: str, output_path: str):
    """
    Process extracted shipping data and match products with ETRM data.
    
    Args:
        extracted_data_path: Path to extracted shipping data (CSV or Excel)
        etrm_data_path: Path to ETRM data (Excel)
        output_path: Path to save results (without extension)
    """
    try:
        # Validate input paths
        if not all(os.path.exists(path) for path in [extracted_data_path, etrm_data_path]):
            missing = [path for path in [extracted_data_path, etrm_data_path] if not os.path.exists(path)]
            raise FileNotFoundError(f"Input files not found: {', '.join(missing)}")

        # Load data - handle both CSV and Excel files
        if extracted_data_path.endswith('.csv'):
            try:
                extracted_df = pd.read_csv(extracted_data_path, encoding='utf-8')
            except UnicodeDecodeError:
                extracted_df = pd.read_csv(extracted_data_path, encoding='latin1')
        elif extracted_data_path.endswith(('.xlsx', '.xls')):
            extracted_df = pd.read_excel(extracted_data_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

        # Initialize matcher
        matcher = ProductMatcher(etrm_data_path)

        # Validate required column exists
        if 'Product Name' not in extracted_df.columns:
            raise ValueError("Input data must contain a 'Product Name' column")

        # Extract product names
        product_names = extracted_df['Product Name'].tolist()
        
        # Match products
        results = matcher.match_products(product_names, batch_mode=True, use_llm=True)
        
        # Merge with original data
        results_df = pd.DataFrame(results)
        final_df = extracted_df.merge(
            results_df, 
            left_on='Product Name', 
            right_on='input_product',
            how='left'
        )
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        matcher.save_results(final_df.to_dict('records'), output_path)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Load API key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Please set your OPENAI_API_KEY environment variable")


    # Paths to your data files - UPDATE THESE TO YOUR ACTUAL PATHS
    extracted_data_path = "ETRM_AI_Matching_RB/data/Extracted_Data.xlsx"  # Update this path
    etrm_data_path = "ETRM_AI_Matching_RB/data/ETRM_Data.xlsx"  # Update this path
    output_path = "ETRM_AI_Matching_RB/results/product_matching_results"  # Update this path

    process_extracted_data(extracted_data_path, etrm_data_path, output_path)
