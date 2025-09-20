"""
Financial Modeling Team - Utilities Package
Common utilities and helper functions for financial modeling
"""

from .financial_utils import FinancialCalculator
from .data_sources import DataSourceManager

__all__ = [
    'FinancialCalculator',
    'DataSourceManager'
]

__version__ = "1.0.0"
__author__ = "Financial Modeling Team"
__description__ = "Utility functions and classes for financial modeling and data management"

# Utility registry
UTILITY_REGISTRY = {
    'calculator': FinancialCalculator,
    'data_manager': DataSourceManager
}

def get_utility_class(utility_name: str):
    """
    Get utility class by name
    
    Args:
        utility_name: Name of the utility
        
    Returns:
        Utility class or None if not found
    """
    return UTILITY_REGISTRY.get(utility_name)

def create_calculator():
    """
    Create a financial calculator instance
    
    Returns:
        FinancialCalculator instance
    """
    return FinancialCalculator()

def create_data_manager():
    """
    Create a data source manager instance
    
    Returns:
        DataSourceManager instance
    """
    return DataSourceManager()

# Common financial constants
FINANCIAL_CONSTANTS = {
    'TRADING_DAYS_PER_YEAR': 252,
    'BUSINESS_DAYS_PER_YEAR': 261,
    'CALENDAR_DAYS_PER_YEAR': 365,
    'DEFAULT_RISK_FREE_RATE': 0.02,
    'DEFAULT_MARKET_RETURN': 0.08,
    'QUARTERS_PER_YEAR': 4,
    'MONTHS_PER_YEAR': 12,
    'WEEKS_PER_YEAR': 52
}

# Common market indices symbols
MARKET_INDICES = {
    'SP500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW_JONES': '^DJI',
    'RUSSELL_2000': '^RUT',
    'VIX': '^VIX',
    'FTSE_100': '^FTSE',
    'NIKKEI': '^N225',
    'DAX': '^GDAXI'
}

# Common currency pairs
CURRENCY_PAIRS = {
    'EUR_USD': 'EURUSD=X',
    'GBP_USD': 'GBPUSD=X',
    'USD_JPY': 'JPY=X',
    'USD_CHF': 'CHF=X',
    'AUD_USD': 'AUDUSD=X',
    'USD_CAD': 'CAD=X'
}

# Common commodities
COMMODITIES = {
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'CRUDE_OIL': 'CL=F',
    'NATURAL_GAS': 'NG=F',
    'COPPER': 'HG=F'
}

def get_symbol(category: str, name: str) -> str:
    """
    Get trading symbol for common instruments
    
    Args:
        category: Category (indices, currencies, commodities)
        name: Instrument name
        
    Returns:
        Trading symbol or None if not found
    """
    symbol_maps = {
        'indices': MARKET_INDICES,
        'currencies': CURRENCY_PAIRS,
        'commodities': COMMODITIES
    }
    
    symbol_map = symbol_maps.get(category, {})
    return symbol_map.get(name.upper())