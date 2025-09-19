import streamlit as st
import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from dotenv import load_dotenv
import openai
from datetime import datetime
import sqlite3
from sqlite3 import Connection
import uuid

# Load environment variables
load_dotenv()

# --- Configuration ---
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    ETRM_DATA_PATH = os.getenv("ETRM_DATA_PATH", "ETRM_Data.xlsx")
    
    # Database path for abbreviations
    DB_PATH = "abbreviations.db"

config = Config()

# Initialize OpenAI
openai.api_key = config.OPENAI_API_KEY
if config.AZURE_OPENAI_ENDPOINT:
    openai.api_base = config.AZURE_OPENAI_ENDPOINT
    openai.api_type = "azure"
    openai.api_version = config.OPENAI_API_VERSION

# --- Database Setup for Abbreviations ---
def init_db(conn: Connection):
    """Initialize the abbreviations database"""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS abbreviations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            abbreviation TEXT UNIQUE,
            full_form TEXT,
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

def get_db_connection() -> Connection:
    """Get a connection to the SQLite database"""
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    init_db(conn)
    return conn

def store_abbreviation(conn: Connection, abbreviation: str, full_form: str, context: str = ""):
    """Store a new abbreviation in the database"""
    try:
        conn.execute(
            "INSERT OR IGNORE INTO abbreviations (abbreviation, full_form, context) VALUES (?, ?, ?)",
            (abbreviation.upper(), full_form, context)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Already exists

def find_abbreviation(conn: Connection, abbreviation: str) -> Optional[str]:
    """Find the full form of an abbreviation"""
    cursor = conn.execute(
        "SELECT full_form FROM abbreviations WHERE abbreviation = ?",
        (abbreviation.upper(),)
    )
    result = cursor.fetchone()
    return result[0] if result else None

# --- Data Loading ---
@st.cache_data
def load_etrm_data():
    """Load ETRM data from Excel file"""
    try:
        df = pd.read_excel(r"C:\Users\RachanaBaldania\OneDrive - RandomTrees\Rachana_Code\ETRM_AI_Matching_RB\V3\data\ETRM_Data.xlsx", sheet_name="Products")
        # Clean up column names
        df.columns = [col.strip().lower() for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error loading ETRM data: {str(e)}")
        return pd.DataFrame()

# --- Agent Definitions ---
class MatchingResult:
    """Class to store matching results"""
    def __init__(self, field_name: str, field_value: str, matched_product: str, 
                 score: float, agent_type: str, sme_comments: str = ""):
        self.field_name = field_name
        self.field_value = field_value
        self.matched_product = matched_product
        self.score = score
        self.agent_type = agent_type
        self.sme_comments = sme_comments
    
    def to_dict(self):
        return {
            "field_name": self.field_name,
            "field_value": self.field_value,
            "matched_product": self.matched_product,
            "score": self.score,
            "agent_type": self.agent_type,
            "sme_comments": self.sme_comments
        }

class BaseMatchingAgent:
    """Base class for all matching agents"""
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.etrm_df = load_etrm_data()
    
    def match(self, field_name: str, field_value: str) -> List[MatchingResult]:
        """Base match method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_product_descriptions(self) -> List[str]:
        """Get all product descriptions from ETRM data"""
        return self.etrm_df['description'].dropna().tolist()

class RuleBasedAgent(BaseMatchingAgent):
    """Agent that uses predefined rules for matching"""
    def __init__(self):
        super().__init__("rule_based")
        # Define rules for common patterns
        self.rules = {
            "ULSD": "ULTRA LOW SULFUR DIESEL",
            "RD99": "RENEWABLE DIESEL",
            "B100": "BIODIESEL 100",
            "B20": "BIODIESEL 20",
            "B5": "BIODIESEL 5",
            "B10": "BIODIESEL 10",
            "B15": "BIODIESEL 15",
            "B50": "BIODIESEL 50",
            "B99": "BIODIESEL 99",
            "LSR": "LIGHT STRAIGHT RUN",
            "HSR": "HEAVY STRAIGHT RUN",
            "VGO": "VACUUM GAS OIL",
            "HVGO": "HEAVY VACUUM GAS OIL",
            "LVGO": "LIGHT VACUUM GAS OIL",
            "LSFO": "LOW SULFUR FUEL OIL",
            "HSFO": "HIGH SULFUR FUEL OIL",
            "MGO": "MARINE GAS OIL",
            "RUL": "REGULAR UNLEADED",
            "PUL": "PREMIUM UNLEADED",
            "MUL": "MID-GRADE UNLEADED",
            "CBOB": "CONVENTIONAL BLENDSTOCK FOR OXYGENATE BLENDING",
            "RBOB": "REFORMULATED BLENDSTOCK FOR OXYGENATE BLENDING",
            "ETOH": "ETHANOL",
            "MTBE": "METHYL TERTIARY BUTYL ETHER",
            "RVP": "REID VAPOR PRESSURE",
            "CARB": "CALIFORNIA AIR RESOURCES BOARD",
            "RFG": "REFORMULATED GASOLINE",
            "VOC": "VOLATILE ORGANIC COMPOUNDS",
            "NOX": "NITROGEN OXIDES",
            "PM": "PARTICULATE MATTER",
            "PPMS": "PARTS PER MILLION SULFUR",
            "UDY": "UNDYED",
            "DYE": "DYED",
            "NR": "NO RIN",
            "RIN": "RENEWABLE IDENTIFICATION NUMBER",
            "HDRD": "HYDROTREATED RENEWABLE DIESEL",
            "FAME": "FATTY ACID METHYL ESTER",
            "CME": "CANOLA METHYL ESTER",
            "SME": "SOYBEAN METHYL ESTER",
            "PME": "PALM METHYL ESTER",
            "RME": "RAPESEED METHYL ESTER",
            "GTL": "GAS TO LIQUIDS",
            "INT": "INTERMEDIATE",
            "GN": "GENERIC",
            "BF": "BRANDED FUEL",
            "SH": "SHELL",
            "BR": "BP",
            "UB": "UNBRANDED",
            "NP": "NON-PREMIXED",
            "MV": "MOTOR VEHICLE",
            "NR": "NON-ROAD",
            "HS": "HIGH SULFUR",
            "LS": "LOW SULFUR",
            "LM": "LIMITED SULFUR",
            "AGO": "AUTOMOTIVE GAS OIL",
            "DHO": "DIESEL HEATING OIL",
            "MFO": "MARINE FUEL OIL",
            "RFO": "RESIDUAL FUEL OIL",
            "FO": "FUEL OIL",
            "HFO": "HEAVY FUEL OIL",
            "LFO": "LIGHT FUEL OIL",
            "GO": "GAS OIL",
            "VGO": "VACUUM GAS OIL",
            "SR": "STRAIGHT RUN",
            "CR": "CRACKED",
            "CAT": "CATALYTIC",
            "HC": "HYDROCARBON",
            "ALK": "ALKYLATE",
            "BTX": "BENZENE, TOLUENE, XYLENE",
            "BZ": "BENZENE",
            "TOL": "TOLUENE",
            "XYL": "XYLENE",
            "PX": "PARA-XYLENE",
            "OX": "ORTHO-XYLENE",
            "MX": "META-XYLENE",
            "C4": "BUTANE/BUTYLENE",
            "C3": "PROPANE/PROPYLENE",
            "C2": "ETHANE/ETHYLENE",
            "C5": "PENTANE",
            "C6": "HEXANE",
            "C7": "HEPTANE",
            "C8": "OCTANE",
            "C9": "NONANE",
            "C10": "DECANE",
            "LPG": "LIQUEFIED PETROLEUM GAS",
            "LNG": "LIQUEFIED NATURAL GAS",
            "CNG": "COMPRESSED NATURAL GAS",
            "LNG": "LIQUEFIED NATURAL GAS",
            "NGL": "NATURAL GAS LIQUIDS",
            "NGL": "NATURAL GASOLINE",
            "CON": "CONDENSATE",
            "WCS": "WESTERN CANADIAN SELECT",
            "WTI": "WEST TEXAS INTERMEDIATE",
            "Brent": "BRENT CRUDE",
            "LLS": "LOUISIANA LIGHT SWEET",
            "ANS": "ALASKA NORTH SLOPE",
        }
    
    def match(self, field_name: str, field_value: str) -> List[MatchingResult]:
        """Match using predefined rules"""
        results = []
        
        # Check if the field value matches any rule
        for abbrev, full_form in self.rules.items():
            if abbrev in field_value.upper():
                # Look for products that contain the full form
                matched_products = self.etrm_df[
                    self.etrm_df['description'].str.contains(full_form, case=False, na=False)
                ]
                
                for _, product in matched_products.iterrows():
                    score = self._calculate_score(field_value, product['description'], abbrev, full_form)
                    results.append(MatchingResult(
                        field_name=field_name,
                        field_value=field_value,
                        matched_product=product['description'],
                        score=score,
                        agent_type=self.agent_type,
                        sme_comments=f"Rule-based match: {abbrev} -> {full_form}"
                    ))
        
        return results[:5]  # Return top 5 results
    
    def _calculate_score(self, field_value: str, product_desc: str, abbrev: str, full_form: str) -> float:
        """Calculate matching score based on rule application"""
        base_score = 0.7  # Base score for rule matches
        
        # Increase score if full form is in product description
        if full_form.lower() in product_desc.lower():
            base_score += 0.2
        
        # Increase score if abbreviation is at the beginning of the field value
        if field_value.upper().startswith(abbrev):
            base_score += 0.1
        
        return min(1.0, base_score)

class AbbreviationAgent(BaseMatchingAgent):
    """Agent that handles abbreviations using a database"""
    def __init__(self, db_conn: Connection):
        super().__init__("abbreviation")
        self.db_conn = db_conn
    
    def match(self, field_name: str, field_value: str) -> List[MatchingResult]:
        """Match using abbreviation database"""
        results = []
        
        # Extract potential abbreviations (words in uppercase, 2-5 characters)
        words = re.findall(r'\b[A-Z]{2,5}\b', field_value.upper())
        
        for abbrev in words:
            full_form = find_abbreviation(self.db_conn, abbrev)
            
            if full_form:
                # Look for products that contain the full form
                matched_products = self.etrm_df[
                    self.etrm_df['description'].str.contains(full_form, case=False, na=False)
                ]
                
                for _, product in matched_products.iterrows():
                    score = self._calculate_score(field_value, product['description'], abbrev, full_form)
                    results.append(MatchingResult(
                        field_name=field_name,
                        field_value=field_value,
                        matched_product=product['description'],
                        score=score,
                        agent_type=self.agent_type,
                        sme_comments=f"Abbreviation match: {abbrev} -> {full_form}"
                    ))
        
        return results[:5]  # Return top 5 results
    
    def _calculate_score(self, field_value: str, product_desc: str, abbrev: str, full_form: str) -> float:
        """Calculate matching score based on abbreviation expansion"""
        base_score = 0.6  # Base score for abbreviation matches
        
        # Increase score if full form is in product description
        if full_form.lower() in product_desc.lower():
            base_score += 0.3
        
        # Increase score if abbreviation is at the beginning of the field value
        if field_value.upper().startswith(abbrev):
            base_score += 0.1
        
        return min(1.0, base_score)

class ExactMatchAgent(BaseMatchingAgent):
    """Agent that performs exact matching"""
    def __init__(self):
        super().__init__("exact")
    
    def match(self, field_name: str, field_value: str) -> List[MatchingResult]:
        """Perform exact matching"""
        results = []
        
        # Look for exact matches in description
        exact_matches = self.etrm_df[
            self.etrm_df['description'].str.lower() == field_value.lower()
        ]
        
        for _, product in exact_matches.iterrows():
            results.append(MatchingResult(
                field_name=field_name,
                field_value=field_value,
                matched_product=product['description'],
                score=1.0,  # Perfect score for exact matches
                agent_type=self.agent_type,
                sme_comments="Exact match found"
            ))
        
        return results

class PartialMatchAgent(BaseMatchingAgent):
    """Agent that performs partial matching"""
    def __init__(self):
        super().__init__("partial")
    
    def match(self, field_name: str, field_value: str) -> List[MatchingResult]:
        """Perform partial matching"""
        results = []
        
        # Look for partial matches in description
        partial_matches = self.etrm_df[
            self.etrm_df['description'].str.contains(field_value, case=False, na=False)
        ]
        
        for _, product in partial_matches.iterrows():
            score = self._calculate_score(field_value, product['description'])
            results.append(MatchingResult(
                field_name=field_name,
                field_value=field_value,
                matched_product=product['description'],
                score=score,
                agent_type=self.agent_type,
                sme_comments="Partial match found"
            ))
        
        # Sort by score and return top 5
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:5]
    
    def _calculate_score(self, field_value: str, product_desc: str) -> float:
        """Calculate matching score based on string similarity"""
        # Simple Jaccard similarity
        field_words = set(field_value.lower().split())
        product_words = set(product_desc.lower().split())
        
        if not field_words or not product_words:
            return 0.0
        
        intersection = field_words.intersection(product_words)
        union = field_words.union(product_words)
        
        return len(intersection) / len(union)

class SemanticMatchAgent(BaseMatchingAgent):
    """Agent that performs semantic matching using LLM"""
    def __init__(self):
        super().__init__("semantic")
    
    def match(self, field_name: str, field_value: str) -> List[MatchingResult]:
        """Perform semantic matching using LLM"""
        results = []
        
        # Get all product descriptions
        product_descriptions = self._get_product_descriptions()
        
        # Use LLM to find semantic matches
        matches = self._find_semantic_matches(field_value, product_descriptions)
        
        for product_desc, score, explanation in matches:
            results.append(MatchingResult(
                field_name=field_name,
                field_value=field_value,
                matched_product=product_desc,
                score=score,
                agent_type=self.agent_type,
                sme_comments=explanation
            ))
        
        return results[:5]  # Return top 5 results
    
    def _find_semantic_matches(self, field_value: str, product_descriptions: List[str]) -> List[Tuple[str, float, str]]:
        """Use LLM to find semantic matches"""
        # For demonstration, we'll use a simple approach
        # In a real implementation, you would use embeddings or fine-tuned models
        
        # This is a simplified version - in practice, you'd use proper embeddings
        matches = []
        
        # Simple keyword-based approach as fallback
        for desc in product_descriptions:
            common_words = set(field_value.lower().split()) & set(desc.lower().split())
            if common_words:
                score = len(common_words) / max(len(field_value.split()), len(desc.split()))
                if score > 0.3:  # Threshold
                    matches.append((desc, score, f"Semantic match based on common keywords: {', '.join(common_words)}"))
        
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]

# --- Orchestrator ---
class MatchingOrchestrator:
    """Orchestrator that coordinates multiple matching agents"""
    def __init__(self, db_conn: Connection):
        self.agents = {
            "rule": RuleBasedAgent(),
            "abbreviation": AbbreviationAgent(db_conn),
            "exact": ExactMatchAgent(),
            "partial": PartialMatchAgent(),
            "semantic": SemanticMatchAgent()
        }
        self.agent_priority = ["rule", "abbreviation", "exact", "partial", "semantic"]
    
    def match_field(self, field_name: str, field_value: str) -> List[MatchingResult]:
        """Match a field using all agents in priority order"""
        all_results = []
        
        for agent_name in self.agent_priority:
            agent = self.agents[agent_name]
            try:
                results = agent.match(field_name, field_value)
                all_results.extend(results)
                
                # If we have high-confidence matches from early agents, we can stop
                if results and any(r.score > 0.8 for r in results) and agent_name in ["rule", "abbreviation", "exact"]:
                    break
                    
            except Exception as e:
                st.warning(f"Agent {agent_name} failed: {str(e)}")
        
        # Sort by score and remove duplicates
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Remove duplicates (keep highest score for each matched product)
        seen_products = set()
        unique_results = []
        
        for result in all_results:
            if result.matched_product not in seen_products:
                seen_products.add(result.matched_product)
                unique_results.append(result)
        
        return unique_results[:5]  # Return top 5 unique results

# --- Streamlit App ---
def main():
    st.title("ETRM Product Matching System")
    st.subheader("Multi-Agent Orchestrator for Product Name Matching")
    
    # Initialize database connection
    db_conn = get_db_connection()
    
    # Initialize orchestrator
    orchestrator = MatchingOrchestrator(db_conn)
    
    # Load ETRM data
    etrm_df = load_etrm_data()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload JSON file", type=["json"])
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load JSON data
            json_data = json.load(uploaded_file)
            
            # Extract field values from JSON
            field_values = extract_fields_from_json(json_data)
            
            # Display extracted fields
            st.header("Extracted Fields")
            for field_name, field_value in field_values.items():
                st.write(f"**{field_name}**: {field_value}")
            
            # Perform matching for each field
            st.header("Matching Results")
            
            for field_name, field_value in field_values.items():
                if field_value:  # Only process non-empty values
                    st.subheader(f"Matching for: {field_name} = '{field_value}'")
                    
                    # Get matches from orchestrator
                    matches = orchestrator.match_field(field_name, field_value)
                    
                    if matches:
                        # Display results in a table
                        match_data = []
                        for match in matches:
                            match_data.append({
                                "Matched Product": match.matched_product,
                                "Score": f"{match.score:.2%}",
                                "Agent": match.agent_type,
                                "SME Comments": match.sme_comments
                            })
                        
                        df = pd.DataFrame(match_data)
                        st.dataframe(df)
                        
                        # Show details for top match
                        top_match = matches[0]
                        st.info(f"**Top Match**: {top_match.matched_product} ({top_match.score:.2%})")
                        st.write(f"**Agent**: {top_match.agent_type}")
                        st.write(f"**Comments**: {top_match.sme_comments}")
                    else:
                        st.warning("No matches found for this field.")
                    
                    st.divider()
            
        except Exception as e:
            st.error(f"Error processing JSON file: {str(e)}")
    else:
        # Show sample JSON structure and instructions
        st.info("Please upload a JSON file to begin matching.")
        
        # Display sample JSON structure
        st.subheader("Expected JSON Structure")
        st.json({
            "auto_filename_used": "sample.pdf",
            "current": {
                "final_results": {
                    "commodity": "ULSD",
                    "hazmat_product": "ULTRA LOW SULFUR DIESEL UNDYED",
                    "product": "RD99C",
                    # Other fields that might contain product names
                }
            }
        })
        
        # Display ETRM data preview
        st.subheader("ETRM Data Preview")
        if not etrm_df.empty:
            st.dataframe(etrm_df[['id_number', 'name', 'description']].head(10))
        else:
            st.warning("Could not load ETRM data. Please check the file path.")

def extract_fields_from_json(json_data: Dict) -> Dict[str, str]:
    """Extract potential product name fields from JSON data"""
    fields = {}
    
    # Common field names that might contain product information
    product_field_names = [
        'commodity', 'product', 'hazmat_product', 'scale_ticket_product',
        'Product', 'ProductName', 'product_name', 'product_code',
        'Hazmat Description', 'product_description', 'material',
        'goods_description', 'item_description', 'material_description'
    ]
    
    def extract_from_dict(data_dict, prefix=""):
        """Recursively extract fields from dictionary"""
        for key, value in data_dict.items():
            if isinstance(value, dict):
                extract_from_dict(value, f"{prefix}{key}.")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        extract_from_dict(item, f"{prefix}{key}[{i}].")
            else:
                if key in product_field_names and value and isinstance(value, str):
                    fields[f"{prefix}{key}"] = value
    
    # Extract from current data
    if 'current' in json_data and 'final_results' in json_data['current']:
        extract_from_dict(json_data['current']['final_results'], "current.final_results.")
    
    # Extract from history
    if 'history' in json_data:
        for i, history_item in enumerate(json_data['history']):
            if 'final_results' in history_item:
                extract_from_dict(history_item['final_results'], f"history[{i}].final_results.")
    
    return fields

# --- Abbreviation Management UI ---
def abbreviation_management_ui(db_conn: Connection):
    """UI for managing abbreviations"""
    st.sidebar.header("Abbreviation Management")
    
    with st.sidebar.expander("Add New Abbreviation"):
        abbrev = st.text_input("Abbreviation")
        full_form = st.text_input("Full Form")
        context = st.text_input("Context (optional)")
        
        if st.button("Add Abbreviation"):
            if abbrev and full_form:
                store_abbreviation(db_conn, abbrev, full_form, context)
                st.success(f"Added abbreviation: {abbrev} -> {full_form}")
            else:
                st.error("Please provide both abbreviation and full form.")
    
    with st.sidebar.expander("View Abbreviations"):
        cursor = db_conn.execute("SELECT abbreviation, full_form, context FROM abbreviations ORDER BY abbreviation")
        abbreviations = cursor.fetchall()
        
        if abbreviations:
            for abbrev, full_form, context in abbreviations:
                st.write(f"**{abbrev}**: {full_form}")
                if context:
                    st.caption(f"Context: {context}")
        else:
            st.info("No abbreviations in database.")

if __name__ == "__main__":
    # Initialize database connection
    db_conn = get_db_connection()
    
    # Run the main app
    main()
    
    # Add abbreviation management UI to sidebar
    abbreviation_management_ui(db_conn)
    
    # Close database connection when done
    db_conn.close()