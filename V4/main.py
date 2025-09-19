import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# --- Configuration ---
ETRM_DATA_PATH = os.getenv("ETRM_DATA_PATH")
CHROMA_COLLECTION_NAME = "product_abbreviations"
CHROMA_DB_PATH = "./chroma_db"

# Sample ETRM data (replace with actual data loading)
etrm_data = pd.DataFrame({
    "Description": [
        "INT NAPH Renewable Naphtha Trading US (8292)",
        "REN Diesel 99 Carbon Neutral",
        "RD99 Renewable Diesel",
        "CNG Compressed Natural Gas",
        "BIO Biofuel Blend 20"
    ],
    "ProductCode": ["8292", "RD99", "RD99C", "CNG01", "BIO20"]
})

# Initialize ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=sentence_transformer_ef)

# Add sample abbreviations to ChromaDB (would normally be pre-populated)
abbreviation_mappings = [
    {"abbreviation": "RD99C", "full_name": "Renewable Diesel 99 Carbon Neutral"},
    {"abbreviation": "INT NAPH", "full_name": "International Naphtha"},
    {"abbreviation": "CNG", "full_name": "Compressed Natural Gas"},
    {"abbreviation": "BIO", "full_name": "Biofuel"}
]

for idx, mapping in enumerate(abbreviation_mappings):
    collection.add(
        documents=[mapping["full_name"]],
        metadatas=[{"abbreviation": mapping["abbreviation"]}],
        ids=[f"id{idx}"]
    )

# --- AutoGen Configuration ---
config_list = [
    {
        "model": os.getenv("OPENAI_DEPLOYMENT_NAME"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_type": "azure",
        "api_version": os.getenv("OPENAI_API_VERSION")
    }
]

llm_config = {
    "config_list": config_list,
    "timeout": 120,
    "temperature": 0
}

# --- Agent Definitions ---
class MatchingResult(TypedDict):
    field_name: str
    field_value: str
    matched_value: str
    matching_type: str
    matching_score: float
    sme_comments: Optional[str]

def rule_match_agent(product_name: str) -> List[MatchingResult]:
    """Rule-based matching agent (checks SME comments first)"""
    # In a real implementation, this would check a database for SME-defined rules
    return []

def abbreviation_match_agent(product_name: str) -> List[MatchingResult]:
    """Abbreviation matching using vector database"""
    results = collection.query(query_texts=[product_name], n_results=5)
    matches = []
    
    for i in range(len(results["ids"][0])):
        full_name = results["documents"][0][i]
        abbreviation = results["metadatas"][0][i]["abbreviation"]
        distance = 1 - results["distances"][0][i]  # Convert to similarity score
        
        matches.append({
            "field_name": "Product Name",
            "field_value": product_name,
            "matched_value": full_name,
            "matching_type": "Abbreviation",
            "matching_score": min(100, distance * 100),  # Scale to percentage
            "sme_comments": f"Matched abbreviation: {abbreviation}"
        })
    
    return matches

def exact_match_agent(product_name: str) -> List[MatchingResult]:
    """Exact string matching"""
    matches = []
    for _, row in etrm_data.iterrows():
        if product_name.lower() == row["Description"].lower():
            matches.append({
                "field_name": "Product Name",
                "field_value": product_name,
                "matched_value": row["Description"],
                "matching_type": "Exact",
                "matching_score": 100,
                "sme_comments": "Exact match found"
            })
    return matches

def partial_match_agent(product_name: str) -> List[MatchingResult]:
    """Partial string matching"""
    matches = []
    for _, row in etrm_data.iterrows():
        if product_name.lower() in row["Description"].lower():
            score = (len(product_name) / len(row["Description"])) * 100
            matches.append({
                "field_name": "Product Name",
                "field_value": product_name,
                "matched_value": row["Description"],
                "matching_type": "Partial",
                "matching_score": min(100, score),
                "sme_comments": f"Partial match ({len(product_name)}/{len(row['Description'])} characters)"
            })
    return sorted(matches, key=lambda x: x["matching_score"], reverse=True)[:5]

def semantic_match_agent(product_name: str) -> List[MatchingResult]:
    """Semantic similarity matching using embeddings"""
    query_embedding = sentence_transformer_ef([product_name])
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    
    matches = []
    for i in range(len(results["ids"][0])):
        full_name = results["documents"][0][i]
        distance = 1 - results["distances"][0][i]  # Convert to similarity score
        
        matches.append({
            "field_name": "Product Name",
            "field_value": product_name,
            "matched_value": full_name,
            "matching_type": "Semantic",
            "matching_score": min(100, distance * 100),  # Scale to percentage
            "sme_comments": "Semantic similarity match"
        })
    
    return matches

# Create matching agents
rule_agent = AssistantAgent(
    name="Rule_Match_Agent",
    system_message="You are an expert in rule-based matching for renewable energy products. Check SME-defined rules first for any product name matches.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

abbreviation_agent = AssistantAgent(
    name="Abbreviation_Match_Agent",
    system_message="You are an expert in abbreviation matching for renewable energy products. Use the vector database to find abbreviations.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

exact_agent = AssistantAgent(
    name="Exact_Match_Agent",
    system_message="You are an expert in exact string matching for renewable energy products.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

partial_agent = AssistantAgent(
    name="Partial_Match_Agent",
    system_message="You are an expert in partial string matching for renewable energy products.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

semantic_agent = AssistantAgent(
    name="Semantic_Match_Agent",
    system_message="You are an expert in semantic similarity matching for renewable energy products.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Create orchestrator agent
orchestrator = AssistantAgent(
    name="Matching_Orchestrator",
    system_message="""You are the Matching Manager Orchestrator for renewable energy products. Your job is to:
1. First check rule-based matches (SME comments)
2. If no rule match, check abbreviation matches
3. If no abbreviation match, check exact matches
4. If no exact match, check partial matches
5. Finally, check semantic matches
Return the top 5 matches from each technique.""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Create user proxy
user_proxy = UserProxyAgent(
    name="User_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
    system_message="A proxy for the user to interact with the matching system.",
    llm_config=llm_config
)

# Register functions
def register_matching_functions(agent):
    @agent.register_for_execution()
    def rule_match(product_name: str) -> List[Dict]:
        return rule_match_agent(product_name)
    
    @agent.register_for_execution()
    def abbreviation_match(product_name: str) -> List[Dict]:
        return abbreviation_match_agent(product_name)
    
    @agent.register_for_execution()
    def exact_match(product_name: str) -> List[Dict]:
        return exact_match_agent(product_name)
    
    @agent.register_for_execution()
    def partial_match(product_name: str) -> List[Dict]:
        return partial_match_agent(product_name)
    
    @agent.register_for_execution()
    def semantic_match(product_name: str) -> List[Dict]:
        return semantic_match_agent(product_name)

register_matching_functions(orchestrator)

# --- Streamlit UI ---
st.title("Renewable Energy Product Matching Manager")
st.markdown("""
This system matches extracted product names with ETRM data descriptions using multiple matching techniques:
1. Rule-based (SME comments)
2. Abbreviation matching
3. Exact matching
4. Partial matching
5. Semantic matching
""")

# Input section
uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
json_data = None

if uploaded_file:
    json_data = json.load(uploaded_file)
    st.success("File uploaded successfully!")
    
    # Extract product name
    try:
        product_name = json_data["extracted_data"]["ShippingDocument"]["ProductDetails"]["Product"]
        st.subheader(f"Extracted Product Name: `{product_name}`")
        
        # Perform matching
        if st.button("Run Matching"):
            with st.spinner("Matching in progress..."):
                # Initialize chat
                user_proxy.initiate_chat(
                    orchestrator,
                    message=f"Match this product name: {product_name}",
                    clear_history=True
                )
                
                # Collect results from all agents
                all_results = []
                
                # Rule match
                rule_results = rule_match_agent(product_name)
                if rule_results:
                    all_results.extend(rule_results)
                    st.success(f"Found {len(rule_results)} rule-based matches")
                else:
                    st.info("No rule-based matches found")
                
                # Abbreviation match
                abbrev_results = abbreviation_match_agent(product_name)
                if abbrev_results:
                    all_results.extend(abbrev_results)
                    st.success(f"Found {len(abbrev_results)} abbreviation matches")
                else:
                    st.info("No abbreviation matches found")
                
                # Exact match
                exact_results = exact_match_agent(product_name)
                if exact_results:
                    all_results.extend(exact_results)
                    st.success(f"Found {len(exact_results)} exact matches")
                else:
                    st.info("No exact matches found")
                
                # Partial match
                partial_results = partial_match_agent(product_name)
                if partial_results:
                    all_results.extend(partial_results)
                    st.success(f"Found {len(partial_results)} partial matches")
                else:
                    st.info("No partial matches found")
                
                # Semantic match
                semantic_results = semantic_match_agent(product_name)
                if semantic_results:
                    all_results.extend(semantic_results)
                    st.success(f"Found {len(semantic_results)} semantic matches")
                else:
                    st.info("No semantic matches found")
                
                # Display results
                if all_results:
                    st.subheader("Matching Results")
                    
                    # Create DataFrame for display
                    results_df = pd.DataFrame(all_results)
                    results_df = results_df.sort_values("matching_score", ascending=False)
                    
                    # Show table
                    st.dataframe(results_df)
                    
                    # Download button
                    st.download_button(
                        label="Download Results as JSON",
                        data=json.dumps(all_results, indent=2),
                        file_name="matching_results.json",
                        mime="application/json"
                    )
                    
                    # Show by matching type
                    st.subheader("Results by Matching Technique")
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Rule", "Abbreviation", "Exact", "Partial", "Semantic"
                    ])
                    
                    with tab1:
                        st.dataframe(pd.DataFrame(rule_results))
                    with tab2:
                        st.dataframe(pd.DataFrame(abbrev_results))
                    with tab3:
                        st.dataframe(pd.DataFrame(exact_results))
                    with tab4:
                        st.dataframe(pd.DataFrame(partial_results))
                    with tab5:
                        st.dataframe(pd.DataFrame(semantic_results))
                else:
                    st.warning("No matches found using any technique")
                    
    except KeyError as e:
        st.error(f"Could not extract product name from JSON: {str(e)}")

# Sample JSON input for testing
with st.expander("Sample JSON Input"):
    st.code("""{
  "extracted_data": {
    "ShippingDocument": {
      "ProductDetails": {
        "Product": "RD99C"
      }
    }
  }
}""")

# Agent status
with st.expander("Agent Status"):
    st.write("**Active Agents:**")
    st.write("- Rule Match Agent")
    st.write("- Abbreviation Match Agent")
    st.write("- Exact Match Agent")
    st.write("- Partial Match Agent")
    st.write("- Semantic Match Agent")
    st.write("- Matching Orchestrator")