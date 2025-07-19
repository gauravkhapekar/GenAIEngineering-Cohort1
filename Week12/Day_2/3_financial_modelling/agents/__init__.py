"""
Financial Modeling Team - Agents Package
Specialized AI agents for financial analysis and modeling
"""

from .data_analyst import DataAnalystAgent
from .risk_assessor import RiskAssessmentAgent
from .portfolio_optimizer import PortfolioOptimizerAgent
from .forecaster import FinancialForecasterAgent
from .report_generator import ReportGeneratorAgent

__all__ = [
    'DataAnalystAgent',
    'RiskAssessmentAgent', 
    'PortfolioOptimizerAgent',
    'FinancialForecasterAgent',
    'ReportGeneratorAgent'
]

__version__ = "1.0.0"
__author__ = "Financial Modeling Team"
__description__ = "Specialized AI agents for comprehensive financial modeling and analysis"

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    'data_analyst': DataAnalystAgent,
    'risk_assessor': RiskAssessmentAgent,
    'portfolio_optimizer': PortfolioOptimizerAgent,
    'forecaster': FinancialForecasterAgent,
    'report_generator': ReportGeneratorAgent
}

def get_agent_class(agent_name: str):
    """
    Get agent class by name
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Agent class or None if not found
    """
    return AGENT_REGISTRY.get(agent_name)

def list_available_agents():
    """
    List all available agent types
    
    Returns:
        List of agent names
    """
    return list(AGENT_REGISTRY.keys())

def create_agent_team(model, agent_types=None):
    """
    Create a team of agents
    
    Args:
        model: The language model to use
        agent_types: List of agent types to create (None for all)
        
    Returns:
        Dictionary of initialized agents
    """
    if agent_types is None:
        agent_types = list(AGENT_REGISTRY.keys())
    
    team = {}
    for agent_type in agent_types:
        if agent_type in AGENT_REGISTRY:
            agent_class = AGENT_REGISTRY[agent_type]
            team[agent_type] = agent_class(model=model)
    
    return team