# tests/test_agents.py
import sys
from pathlib import Path
import pytest

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def test_agent_imports():
    """Test that all agents can be imported properly"""
    from app.agents import (
        BaseMatchingAgent,
        ExactMatchAgent,
        PartialMatchAgent,
        AbbreviationMatchAgent,
        SemanticMatchAgent,
        AISemanticAgent
    )
    assert True  # If we get here, imports worked

def test_manager_import():
    """Test that MatchingManager can be imported"""
    from ETRM_AI_Matching_RB.V3.app.managers.matching_orchestrator import MatchingManager
    assert True

if __name__ == "__main__":
    pytest.main([__file__])