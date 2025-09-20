"""
Data Sources Manager - Configuration and Management of Financial Data Sources
Supporting the Financial Modeling Team
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataSourceManager:
    """
    Manages various financial data sources and their configurations
    """
    
    def __init__(self):
        self.data_sources = self._initialize_data_sources()
        self.rate_limits = self._initialize_rate_limits()
        self.cache_settings = self._initialize_cache_settings()
        
    def _initialize_data_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available data sources and their configurations"""
        return {
            "yahoo_finance": {
                "name": "Yahoo Finance",
                "type": "free",
                "url": "https://finance.yahoo.com",
                "api_endpoint": "https://query1.finance.yahoo.com/v8/finance/chart/",
                "features": [
                    "real_time_quotes",
                    "historical_data",
                    "fundamental_data",
                    "options_data",
                    "mutual_funds",
                    "etfs"
                ],
                "rate_limit": "2000_requests_per_hour",
                "data_quality": 0.85,
                "reliability": 0.90,
                "delay_minutes": 15,  # Market data delay
                "supported_markets": ["US", "EU", "ASIA"]
            },
            
            "alpha_vantage": {
                "name": "Alpha Vantage",
                "type": "freemium",
                "url": "https://www.alphavantage.co",
                "api_endpoint": "https://www.alphavantage.co/query",
                "features": [
                    "intraday_data",
                    "daily_data",
                    "forex_data",
                    "crypto_data",
                    "technical_indicators",
                    "fundamental_data"
                ],
                "rate_limit": "5_requests_per_minute_free",
                "data_quality": 0.90,
                "reliability": 0.95,
                "delay_minutes": 0,  # Real-time for premium
                "supported_markets": ["US", "GLOBAL"]
            },
            
            "quandl": {
                "name": "Quandl (Nasdaq Data Link)",
                "type": "premium",
                "url": "https://data.nasdaq.com",
                "api_endpoint": "https://data.nasdaq.com/api/v3/",
                "features": [
                    "economic_data",
                    "commodity_data",
                    "fixed_income",
                    "alternative_data",
                    "fundamental_data"
                ],
                "rate_limit": "unlimited_premium",
                "data_quality": 0.95,
                "reliability": 0.98,
                "delay_minutes": 0,
                "supported_markets": ["GLOBAL"]
            },
            
            "fred": {
                "name": "Federal Reserve Economic Data",
                "type": "free",
                "url": "https://fred.stlouisfed.org",
                "api_endpoint": "https://api.stlouisfed.org/fred/",
                "features": [
                    "economic_indicators",
                    "interest_rates",
                    "inflation_data",
                    "employment_data",
                    "gdp_data"
                ],
                "rate_limit": "120_requests_per_minute",
                "data_quality": 0.98,
                "reliability": 0.99,
                "delay_minutes": 0,
                "supported_markets": ["US"]
            },
            
            "bloomberg": {
                "name": "Bloomberg Terminal/API",
                "type": "enterprise",
                "url": "https://www.bloomberg.com",
                "api_endpoint": "bloomberg_api",
                "features": [
                    "real_time_data",
                    "historical_data",
                    "fundamental_data",
                    "fixed_income",
                    "derivatives",
                    "news_analytics"
                ],
                "rate_limit": "high_volume",
                "data_quality": 0.99,
                "reliability": 0.99,
                "delay_minutes": 0,
                "supported_markets": ["GLOBAL"]
            },
            
            "refinitiv": {
                "name": "Refinitiv (formerly Thomson Reuters)",
                "type": "enterprise",
                "url": "https://www.refinitiv.com",
                "api_endpoint": "refinitiv_api",
                "features": [
                    "real_time_data",
                    "fundamental_data",
                    "news_sentiment",
                    "esg_data",
                    "fixed_income",
                    "commodities"
                ],
                "rate_limit": "high_volume",
                "data_quality": 0.98,
                "reliability": 0.98,
                "delay_minutes": 0,
                "supported_markets": ["GLOBAL"]
            },
            
            "iex_cloud": {
                "name": "IEX Cloud",
                "type": "freemium",
                "url": "https://iexcloud.io",
                "api_endpoint": "https://cloud.iexapis.com/stable/",
                "features": [
                    "real_time_quotes",
                    "historical_data",
                    "fundamental_data",
                    "news_data",
                    "options_data"
                ],
                "rate_limit": "100_requests_per_second_premium",
                "data_quality": 0.92,
                "reliability": 0.96,
                "delay_minutes": 0,
                "supported_markets": ["US"]
            }
        }
    
    def _initialize_rate_limits(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rate limiting configurations"""
        return {
            "yahoo_finance": {
                "requests_per_second": 10,
                "requests_per_minute": 100,
                "requests_per_hour": 2000,
                "burst_limit": 50,
                "cooldown_seconds": 1
            },
            "alpha_vantage": {
                "requests_per_second": 0.2,  # 5 per minute
                "requests_per_minute": 5,
                "requests_per_hour": 300,
                "burst_limit": 5,
                "cooldown_seconds": 12
            },
            "fred": {
                "requests_per_second": 2,
                "requests_per_minute": 120,
                "requests_per_hour": 7200,
                "burst_limit": 10,
                "cooldown_seconds": 0.5
            },
            "iex_cloud": {
                "requests_per_second": 100,
                "requests_per_minute": 6000,
                "requests_per_hour": 360000,
                "burst_limit": 500,
                "cooldown_seconds": 0.01
            }
        }
    
    def _initialize_cache_settings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize caching configurations for different data types"""
        return {
            "real_time_quotes": {
                "ttl_seconds": 60,  # 1 minute cache
                "max_entries": 1000,
                "refresh_threshold": 30
            },
            "intraday_data": {
                "ttl_seconds": 300,  # 5 minute cache
                "max_entries": 500,
                "refresh_threshold": 150
            },
            "daily_data": {
                "ttl_seconds": 3600,  # 1 hour cache
                "max_entries": 200,
                "refresh_threshold": 1800
            },
            "fundamental_data": {
                "ttl_seconds": 86400,  # 24 hour cache
                "max_entries": 100,
                "refresh_threshold": 43200
            },
            "economic_data": {
                "ttl_seconds": 3600,  # 1 hour cache
                "max_entries": 50,
                "refresh_threshold": 1800
            }
        }
    
    def get_data_source_info(self, source_name: str) -> Dict[str, Any]:
        """Get information about a specific data source"""
        return self.data_sources.get(source_name, {})
    
    def list_available_sources(self) -> List[str]:
        """List all available data sources"""
        return list(self.data_sources.keys())
    
    def get_sources_by_type(self, source_type: str) -> List[str]:
        """Get data sources by type (free, freemium, premium, enterprise)"""
        return [
            name for name, config in self.data_sources.items()
            if config.get("type") == source_type
        ]
    
    def get_sources_with_feature(self, feature: str) -> List[str]:
        """Get data sources that support a specific feature"""
        sources = []
        for name, config in self.data_sources.items():
            if feature in config.get("features", []):
                sources.append(name)
        return sources
    
    def get_rate_limit_info(self, source_name: str) -> Dict[str, Any]:
        """Get rate limiting information for a data source"""
        return self.rate_limits.get(source_name, {})
    
    def get_cache_settings(self, data_type: str) -> Dict[str, Any]:
        """Get cache settings for a specific data type"""
        return self.cache_settings.get(data_type, {})
    
    def recommend_data_source(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend the best data sources based on requirements
        
        Args:
            requirements: Dictionary with requirements like:
                - features: List of required features
                - budget: "free", "low", "medium", "high"
                - markets: List of required markets
                - data_quality_min: Minimum data quality (0-1)
                - real_time: Boolean for real-time requirement
        
        Returns:
            List of recommended sources with scores
        """
        recommendations = []
        
        required_features = requirements.get("features", [])
        budget = requirements.get("budget", "free")
        required_markets = requirements.get("markets", [])
        min_quality = requirements.get("data_quality_min", 0.8)
        real_time_needed = requirements.get("real_time", False)
        
        # Budget mapping
        budget_mapping = {
            "free": ["free"],
            "low": ["free", "freemium"],
            "medium": ["free", "freemium", "premium"],
            "high": ["free", "freemium", "premium", "enterprise"]
        }
        
        allowed_types = budget_mapping.get(budget, ["free"])
        
        for source_name, source_config in self.data_sources.items():
            score = 0
            reasons = []
            
            # Check budget compatibility
            if source_config.get("type") not in allowed_types:
                continue
            
            # Check data quality
            if source_config.get("data_quality", 0) < min_quality:
                continue
            
            # Check real-time requirement
            if real_time_needed and source_config.get("delay_minutes", 0) > 5:
                continue
            
            # Score based on features
            source_features = source_config.get("features", [])
            feature_score = len(set(required_features) & set(source_features))
            if required_features:
                feature_score = feature_score / len(required_features)
            else:
                feature_score = 1.0
            
            score += feature_score * 40  # 40% weight for features
            
            if feature_score > 0.8:
                reasons.append("Supports most required features")
            
            # Score based on data quality
            quality_score = source_config.get("data_quality", 0)
            score += quality_score * 25  # 25% weight for quality
            
            if quality_score > 0.95:
                reasons.append("High data quality")
            
            # Score based on reliability
            reliability_score = source_config.get("reliability", 0)
            score += reliability_score * 20  # 20% weight for reliability
            
            if reliability_score > 0.95:
                reasons.append("High reliability")
            
            # Score based on market support
            supported_markets = source_config.get("supported_markets", [])
            if required_markets:
                market_score = len(set(required_markets) & set(supported_markets)) / len(required_markets)
            else:
                market_score = 1.0
            
            score += market_score * 15  # 15% weight for market coverage
            
            if market_score == 1.0:
                reasons.append("Supports all required markets")
            
            # Bonus for free sources
            if source_config.get("type") == "free":
                score += 5
                reasons.append("Free to use")
            
            recommendations.append({
                "source": source_name,
                "score": round(score, 2),
                "reasons": reasons,
                "config": source_config
            })
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations
    
    def create_data_source_portfolio(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a portfolio of data sources for different needs
        
        Args:
            requirements: Requirements dictionary
            
        Returns:
            Dictionary with recommended source portfolio
        """
        portfolio = {
            "primary_source": None,
            "backup_sources": [],
            "specialized_sources": {},
            "cost_estimate": "N/A"
        }
        
        # Get general recommendations
        recommendations = self.recommend_data_source(requirements)
        
        if recommendations:
            portfolio["primary_source"] = recommendations[0]
            
            # Add backup sources
            if len(recommendations) > 1:
                portfolio["backup_sources"] = recommendations[1:3]
        
        # Get specialized sources for specific features
        required_features = requirements.get("features", [])
        
        for feature in required_features:
            specialized = self.get_sources_with_feature(feature)
            if specialized:
                # Get the best source for this feature
                feature_sources = [
                    rec for rec in recommendations 
                    if rec["source"] in specialized
                ]
                if feature_sources:
                    portfolio["specialized_sources"][feature] = feature_sources[0]
        
        # Estimate cost
        primary_type = portfolio["primary_source"]["config"]["type"] if portfolio["primary_source"] else "free"
        cost_mapping = {
            "free": "Free",
            "freemium": "$0-100/month",
            "premium": "$100-1000/month", 
            "enterprise": "$1000+/month"
        }
        portfolio["cost_estimate"] = cost_mapping.get(primary_type, "Unknown")
        
        return portfolio
    
    def validate_data_source_access(self, source_name: str) -> Dict[str, Any]:
        """
        Validate access to a data source (simulated)
        
        Args:
            source_name: Name of the data source
            
        Returns:
            Validation results
        """
        if source_name not in self.data_sources:
            return {
                "valid": False,
                "error": "Data source not found",
                "source": source_name
            }
        
        source_config = self.data_sources[source_name]
        
        # Simulate validation
        validation_result = {
            "valid": True,
            "source": source_name,
            "connection_status": "connected",
            "last_check": datetime.now().isoformat(),
            "rate_limit_status": "normal",
            "data_freshness": "current"
        }
        
        # Simulate some potential issues
        import random
        if random.random() < 0.1:  # 10% chance of issues
            issues = [
                "rate_limit_exceeded",
                "connection_timeout", 
                "api_key_invalid",
                "service_unavailable"
            ]
            issue = random.choice(issues)
            validation_result["valid"] = False
            validation_result["error"] = issue
            validation_result["connection_status"] = "error"
        
        return validation_result
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all configured data sources"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "total_sources": len(self.data_sources),
            "by_type": {},
            "by_status": {"available": 0, "unavailable": 0},
            "source_details": {}
        }
        
        # Count by type
        for source_config in self.data_sources.values():
            source_type = source_config.get("type", "unknown")
            status["by_type"][source_type] = status["by_type"].get(source_type, 0) + 1
        
        # Check each source
        for source_name in self.data_sources.keys():
            validation = self.validate_data_source_access(source_name)
            
            if validation["valid"]:
                status["by_status"]["available"] += 1
            else:
                status["by_status"]["unavailable"] += 1
            
            status["source_details"][source_name] = validation
        
        return status
    
    def generate_data_source_report(self) -> Dict[str, Any]:
        """Generate a comprehensive data source report"""
        report = {
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_sources": len(self.data_sources),
                "free_sources": len(self.get_sources_by_type("free")),
                "premium_sources": len(self.get_sources_by_type("premium")),
                "enterprise_sources": len(self.get_sources_by_type("enterprise"))
            },
            "feature_coverage": self._analyze_feature_coverage(),
            "quality_analysis": self._analyze_data_quality(),
            "recommendations": self._generate_general_recommendations(),
            "cost_analysis": self._analyze_costs()
        }
        
        return report
    
    def _analyze_feature_coverage(self) -> Dict[str, Any]:
        """Analyze feature coverage across data sources"""
        all_features = set()
        feature_sources = {}
        
        for source_config in self.data_sources.values():
            features = source_config.get("features", [])
            all_features.update(features)
            
            for feature in features:
                if feature not in feature_sources:
                    feature_sources[feature] = []
                feature_sources[feature].append(source_config["name"])
        
        return {
            "total_features": len(all_features),
            "features_list": sorted(list(all_features)),
            "feature_sources": feature_sources,
            "redundancy": {
                feature: len(sources) 
                for feature, sources in feature_sources.items()
            }
        }
    
    def _analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality across sources"""
        qualities = [
            config.get("data_quality", 0) 
            for config in self.data_sources.values()
        ]
        
        return {
            "average_quality": np.mean(qualities),
            "min_quality": np.min(qualities),
            "max_quality": np.max(qualities),
            "high_quality_sources": len([q for q in qualities if q > 0.95]),
            "medium_quality_sources": len([q for q in qualities if 0.85 <= q <= 0.95]),
            "low_quality_sources": len([q for q in qualities if q < 0.85])
        }
    
    def _generate_general_recommendations(self) -> List[str]:
        """Generate general recommendations for data source usage"""
        return [
            "Use multiple data sources for critical applications",
            "Implement proper caching to reduce API calls",
            "Monitor rate limits and implement backoff strategies",
            "Regularly validate data source availability",
            "Consider data quality when choosing sources",
            "Have backup sources for high-availability requirements",
            "Implement proper error handling and fallback mechanisms"
        ]
    
    def _analyze_costs(self) -> Dict[str, Any]:
        """Analyze cost structure of data sources"""
        type_counts = {}
        for source_config in self.data_sources.values():
            source_type = source_config.get("type", "unknown")
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
        
        return {
            "free_percentage": (type_counts.get("free", 0) / len(self.data_sources)) * 100,
            "paid_percentage": ((len(self.data_sources) - type_counts.get("free", 0)) / len(self.data_sources)) * 100,
            "type_breakdown": type_counts,
            "cost_optimization_tips": [
                "Start with free sources for basic needs",
                "Upgrade to premium only when necessary",
                "Consider freemium tiers for testing",
                "Negotiate enterprise deals for high volume"
            ]
        }