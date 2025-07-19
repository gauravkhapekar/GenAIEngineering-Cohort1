"""
Configuration Settings for Financial Modeling Team
Customize these settings based on your environment and requirements
"""

import os
from typing import Dict, List, Any

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# OpenAI Model Settings
MODEL_CONFIG = {
    "default_model": "gpt-4",
    "fallback_model": "gpt-3.5-turbo",
    "temperature": 0.1,  # Lower temperature for more consistent financial analysis
    "max_tokens": 2000,
    "timeout": 60  # seconds
}

# Model API Keys (set via environment variables)
API_KEYS = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "alpha_vantage_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "quandl_key": os.getenv("QUANDL_API_KEY"),
    "fred_key": os.getenv("FRED_API_KEY")
}

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

# Agent-specific settings
AGENT_CONFIG = {
    "data_analyst": {
        "enabled": True,
        "cache_results": True,
        "cache_duration_hours": 1,
        "max_symbols_per_request": 10
    },
    "risk_assessor": {
        "enabled": True,
        "var_confidence_levels": [0.90, 0.95, 0.99],
        "stress_test_scenarios": 5,
        "monte_carlo_simulations": 1000
    },
    "portfolio_optimizer": {
        "enabled": True,
        "optimization_methods": ["mean_variance", "risk_parity", "equal_weight"],
        "efficient_frontier_points": 50,
        "constraint_short_selling": False
    },
    "forecaster": {
        "enabled": True,
        "forecast_horizons": ["1M", "3M", "6M", "1Y"],
        "ensemble_methods": ["arima", "garch", "lstm", "technical"],
        "confidence_intervals": True
    },
    "report_generator": {
        "enabled": True,
        "default_format": "json",
        "include_charts": True,
        "executive_summary": True
    }
}

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# Preferred data sources in order of priority
DATA_SOURCE_PRIORITY = [
    "yahoo_finance",  # Free, reliable for basic data
    "alpha_vantage",  # Good for technical indicators
    "fred",           # Economic data
    "iex_cloud"       # Alternative for US markets
]

# Data source specific settings
DATA_SOURCE_CONFIG = {
    "yahoo_finance": {
        "rate_limit_requests_per_second": 5,
        "retry_attempts": 3,
        "timeout_seconds": 30
    },
    "alpha_vantage": {
        "rate_limit_requests_per_minute": 5,  # Free tier limit
        "premium_tier": False,
        "retry_attempts": 2
    },
    "fred": {
        "rate_limit_requests_per_minute": 120,
        "timeout_seconds": 15
    }
}

# =============================================================================
# FINANCIAL CALCULATION SETTINGS
# =============================================================================

FINANCIAL_CONFIG = {
    "risk_free_rate": 0.02,  # 2% default risk-free rate
    "trading_days_per_year": 252,
    "confidence_levels": [0.90, 0.95, 0.99],
    "default_benchmark": "^GSPC",  # S&P 500
    "currency": "USD"
}

# Portfolio constraints
PORTFOLIO_CONSTRAINTS = {
    "max_position_size": 0.25,  # 25% maximum per position
    "min_position_size": 0.01,  # 1% minimum per position
    "max_sector_concentration": 0.40,  # 40% maximum per sector
    "allow_short_selling": False,
    "rebalancing_threshold": 0.05  # 5% drift threshold
}

# =============================================================================
# RISK MANAGEMENT SETTINGS
# =============================================================================

RISK_CONFIG = {
    "var_methods": ["historical", "parametric", "monte_carlo"],
    "stress_test_scenarios": [
        "2008_financial_crisis",
        "2020_covid_crash", 
        "2000_dotcom_bubble",
        "interest_rate_shock",
        "sector_rotation"
    ],
    "risk_tolerance_levels": {
        "conservative": {"max_volatility": 0.10, "max_drawdown": 0.05},
        "moderate": {"max_volatility": 0.15, "max_drawdown": 0.10},
        "aggressive": {"max_volatility": 0.25, "max_drawdown": 0.20}
    },
    "alert_thresholds": {
        "var_breach": 1.2,  # 20% above expected VaR
        "correlation_spike": 0.8,  # Average correlation above 80%
        "concentration_limit": 0.3  # Single position above 30%
    }
}

# =============================================================================
# CACHING AND PERFORMANCE
# =============================================================================

CACHE_CONFIG = {
    "enabled": True,
    "backend": "memory",  # Options: memory, redis, file
    "default_ttl_seconds": 3600,  # 1 hour
    "max_cache_size_mb": 100,
    "cache_by_data_type": {
        "real_time_quotes": 60,      # 1 minute
        "daily_data": 3600,          # 1 hour  
        "fundamental_data": 86400,   # 24 hours
        "economic_data": 3600        # 1 hour
    }
}

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_enabled": True,
    "file_path": "logs/financial_modeling_team.log",
    "console_enabled": True,
    "max_file_size_mb": 10,
    "backup_count": 5
}

# Monitoring and alerting
MONITORING_CONFIG = {
    "enabled": True,
    "metrics_collection": True,
    "alert_on_errors": True,
    "performance_tracking": True,
    "export_metrics": False  # Set to True to export to external systems
}

# =============================================================================
# OUTPUT AND REPORTING
# =============================================================================

OUTPUT_CONFIG = {
    "default_output_dir": "data/outputs",
    "report_formats": ["json", "html", "markdown"],
    "include_charts": True,
    "chart_format": "png",
    "chart_dpi": 300,
    "save_intermediate_results": True
}

# =============================================================================
# DEVELOPMENT AND TESTING
# =============================================================================

DEV_CONFIG = {
    "debug_mode": False,
    "use_sample_data": False,  # Use sample data instead of live APIs
    "mock_api_calls": False,   # Mock API calls for testing
    "verbose_logging": False,
    "profiling_enabled": False
}

# =============================================================================
# SECURITY SETTINGS
# =============================================================================

SECURITY_CONFIG = {
    "encrypt_api_keys": True,
    "mask_sensitive_data_in_logs": True,
    "validate_input_data": True,
    "sanitize_outputs": True,
    "rate_limit_protection": True
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(section: str) -> Dict[str, Any]:
    """Get configuration for a specific section"""
    config_map = {
        "model": MODEL_CONFIG,
        "agents": AGENT_CONFIG,
        "data_sources": DATA_SOURCE_CONFIG,
        "financial": FINANCIAL_CONFIG,
        "risk": RISK_CONFIG,
        "cache": CACHE_CONFIG,
        "logging": LOGGING_CONFIG,
        "output": OUTPUT_CONFIG,
        "dev": DEV_CONFIG,
        "security": SECURITY_CONFIG
    }
    return config_map.get(section, {})

def validate_config() -> Dict[str, List[str]]:
    """Validate configuration settings"""
    issues = {
        "errors": [],
        "warnings": []
    }
    
    # Check API keys
    if not API_KEYS.get("openai_api_key"):
        issues["errors"].append("OpenAI API key not set")
    
    # Check data source settings
    for source in DATA_SOURCE_PRIORITY:
        if source not in DATA_SOURCE_CONFIG:
            issues["warnings"].append(f"No configuration for data source: {source}")
    
    # Check financial settings
    if FINANCIAL_CONFIG["risk_free_rate"] < 0:
        issues["errors"].append("Risk-free rate cannot be negative")
    
    if FINANCIAL_CONFIG["risk_free_rate"] > 0.1:
        issues["warnings"].append("Risk-free rate seems unusually high")
    
    # Check portfolio constraints
    if PORTFOLIO_CONSTRAINTS["max_position_size"] > 1.0:
        issues["errors"].append("Maximum position size cannot exceed 100%")
    
    if PORTFOLIO_CONSTRAINTS["min_position_size"] >= PORTFOLIO_CONSTRAINTS["max_position_size"]:
        issues["errors"].append("Minimum position size must be less than maximum")
    
    return issues

def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment"""
    return {
        "python_version": os.sys.version,
        "working_directory": os.getcwd(),
        "environment_variables": {
            "OPENAI_API_KEY": "Set" if os.getenv("OPENAI_API_KEY") else "Not Set",
            "ALPHA_VANTAGE_API_KEY": "Set" if os.getenv("ALPHA_VANTAGE_API_KEY") else "Not Set",
            "PYTHONPATH": os.getenv("PYTHONPATH", "Not Set")
        },
        "config_validation": validate_config()
    }

# Initialize configuration on import
if __name__ == "__main__":
    # Configuration validation when run directly
    print("Financial Modeling Team Configuration")
    print("=" * 50)
    
    validation = validate_config()
    
    if validation["errors"]:
        print("❌ Configuration Errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    if validation["warnings"]:
        print("⚠️  Configuration Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    if not validation["errors"] and not validation["warnings"]:
        print("✅ Configuration is valid!")
    
    print("\nEnvironment Info:")
    env_info = get_environment_info()
    print(f"Python Version: {env_info['python_version']}")
    print(f"Working Directory: {env_info['working_directory']}")
    print(f"OpenAI API Key: {env_info['environment_variables']['OPENAI_API_KEY']}")