from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from dotenv import load_dotenv
import os
import pandas as pd
from typing import Dict, Any, Optional
from  ..models.schemas import MatchingResponse
from V4.agents.rule_agent import RuleAgent
from  V4.agents.rule_agent import RuleAgent
from  V4.agents.abbreviation_agent import AbbreviationAgent
from  V4.agents.exact_agent import ExactAgent
from  V4.agents.partial_agent import PartialAgent
from  V4.agents.semantic_agent import SemanticAgent

load_dotenv()

class MatchingManagerOrchestrator:
    def __init__(self):
        # Load ETRM data
        self.etrm_data = pd.read_excel(r"C:/Users/RachanaBaldania/OneDrive - RandomTrees/Rachana_Code/ETRM_AI_Matching_RB/data/ETRM_Data.xlsx")
        
        # Initialize agents
        self.rule_agent = RuleAgent(self.etrm_data)
        self.abbreviation_agent = AbbreviationAgent(self.etrm_data)
        self.exact_agent = ExactAgent(self.etrm_data)
        self.partial_agent = PartialAgent(self.etrm_data)
        self.semantic_agent = SemanticAgent(self.etrm_data)
        
        # Agent priority order
        self.agent_priority = [
            self.rule_agent,
            self.abbreviation_agent,
            self.exact_agent,
            self.partial_agent,
            self.semantic_agent
        ]
    
    def match(self, extracted_value: str, field_name: str = "product_name") -> MatchingResponse:
        """Orchestrate the matching process across all agents"""
        best_match = None
        
        for agent in self.agent_priority:
            match_result = agent.match(extracted_value, field_name)
            
            if match_result and (best_match is None or match_result["confidence"] > best_match["confidence"]):
                best_match = match_result
                
                # If we have a perfect match, we can stop early
                if match_result["confidence"] >= 1.0:
                    break
        
        if not best_match:
            return MatchingResponse(
                extracted_value=extracted_value,
                matched_value="",
                matching_type="no_match",
                matching_score=0.0,
                agent_used="none",
                confidence=0.0
            )
        
        return MatchingResponse(**best_match)