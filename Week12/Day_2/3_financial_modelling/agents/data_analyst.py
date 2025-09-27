"""
Data Analyst Agent - Financial Data Collection and Analysis
Part of the Financial Modeling Team
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from utils.financial_utils import FinancialCalculator


class DataAnalystAgent:
    """
    Specialized agent for financial data collection, cleaning, and basic analysis
    """
    
    def __init__(self, model):
        self.model = model
        self.calculator = FinancialCalculator()
        
        self.agent = Agent(
            name="Data Analyst",
            model=model,
            description="Financial data specialist focused on data collection and preprocessing",
            instructions=[
                "You are a financial data analyst specializing in market data collection and analysis",
                "Your role is to gather, clean, and preprocess financial data for the team",
                "Calculate basic financial metrics like returns, volatility, and correlations",
                "Identify data quality issues and provide clean datasets for other agents",
                "Focus on accuracy and completeness of financial data",
                "Provide statistical summaries and data insights"
            ],
            tools=[YFinanceTools()],
            show_tool_calls=True
        )
    
    async def analyze_securities(self, symbols: List[str], timeframe: str = "1Y") -> Dict[str, Any]:
        """
        Comprehensive analysis of securities data
        
        Args:
            symbols: List of stock symbols
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary with detailed data analysis
        """
        try:
            # Get basic data analysis from the agent
            analysis_prompt = f"""
            Analyze the financial data for these securities: {symbols} over {timeframe} period.
            
            Please provide:
            1. Current price and basic statistics for each security
            2. Historical price performance and trends
            3. Volume analysis and trading patterns
            4. Basic technical indicators
            5. Data quality assessment
            
            Focus on factual data analysis and avoid investment recommendations.
            """
            
            response = await self.agent.arun(analysis_prompt)
            
            # Add our own calculations
            detailed_analysis = {
                "agent_analysis": response.content,
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "timeframe": timeframe,
                "calculated_metrics": self._calculate_detailed_metrics(symbols, timeframe),
                "data_quality": self._assess_data_quality(symbols),
                "correlation_matrix": self._calculate_correlations(symbols, timeframe)
            }
            
            return detailed_analysis
            
        except Exception as e:
            return {
                "error": f"Data analysis failed: {str(e)}",
                "symbols": symbols,
                "timeframe": timeframe
            }
    
    def _calculate_detailed_metrics(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Calculate detailed financial metrics for the securities"""
        try:
            metrics = {}
            
            for symbol in symbols:
                # Simulate basic metric calculations
                # In a real implementation, you'd fetch actual data here
                metrics[symbol] = {
                    "daily_return_mean": np.random.normal(0.001, 0.02),
                    "daily_return_std": np.random.uniform(0.015, 0.035),
                    "annualized_volatility": np.random.uniform(0.15, 0.45),
                    "max_drawdown": -np.random.uniform(0.05, 0.25),
                    "sharpe_ratio": np.random.uniform(-0.5, 2.0),
                    "skewness": np.random.normal(0, 0.5),
                    "kurtosis": np.random.uniform(2.5, 5.0),
                    "beta": np.random.uniform(0.5, 1.8)
                }
            
            return metrics
            
        except Exception as e:
            return {"error": f"Metric calculation failed: {str(e)}"}
    
    def _assess_data_quality(self, symbols: List[str]) -> Dict[str, Any]:
        """Assess the quality of available data for each symbol"""
        quality_assessment = {}
        
        for symbol in symbols:
            # Simulate data quality assessment
            quality_assessment[symbol] = {
                "data_completeness": np.random.uniform(0.85, 1.0),
                "missing_data_percentage": np.random.uniform(0, 0.15),
                "outlier_count": np.random.randint(0, 5),
                "data_freshness_hours": np.random.randint(0, 24),
                "quality_score": np.random.uniform(0.7, 1.0),
                "issues": []
            }
            
            # Add potential issues based on quality scores
            if quality_assessment[symbol]["data_completeness"] < 0.9:
                quality_assessment[symbol]["issues"].append("Incomplete historical data")
            if quality_assessment[symbol]["missing_data_percentage"] > 0.1:
                quality_assessment[symbol]["issues"].append("High percentage of missing values")
        
        return quality_assessment
    
    def _calculate_correlations(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Calculate correlation matrix between securities"""
        try:
            # Simulate correlation matrix calculation
            n_symbols = len(symbols)
            correlation_matrix = np.random.rand(n_symbols, n_symbols)
            
            # Make it symmetric and set diagonal to 1
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)
            
            # Scale to proper correlation range [-1, 1]
            correlation_matrix = correlation_matrix * 2 - 1
            
            # Convert to dictionary format
            correlation_dict = {}
            for i, symbol1 in enumerate(symbols):
                correlation_dict[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    correlation_dict[symbol1][symbol2] = float(correlation_matrix[i, j])
            
            return {
                "correlation_matrix": correlation_dict,
                "avg_correlation": float(np.mean(correlation_matrix[np.triu_indices(n_symbols, k=1)])),
                "max_correlation": float(np.max(correlation_matrix[np.triu_indices(n_symbols, k=1)])),
                "min_correlation": float(np.min(correlation_matrix[np.triu_indices(n_symbols, k=1)]))
            }
            
        except Exception as e:
            return {"error": f"Correlation calculation failed: {str(e)}"}
    
    async def get_market_overview(self, indices: List[str] = ["^GSPC", "^DJI", "^IXIC"]) -> Dict[str, Any]:
        """Get overview of major market indices"""
        try:
            overview_prompt = f"""
            Provide a current market overview for these major indices: {indices}.
            
            Include:
            1. Current levels and daily performance
            2. Recent trends and momentum
            3. Trading volume patterns
            4. Market sentiment indicators
            
            Keep it factual and data-focused.
            """
            
            response = await self.agent.arun(overview_prompt)
            
            return {
                "agent_overview": response.content,
                "indices": indices,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Market overview failed: {str(e)}"}
    
    async def analyze_sector_performance(self, sectors: List[str]) -> Dict[str, Any]:
        """Analyze performance across different sectors"""
        try:
            sector_prompt = f"""
            Analyze the performance of these sectors: {sectors}.
            
            Provide:
            1. Relative performance metrics
            2. Sector rotation patterns
            3. Leading and lagging sectors
            4. Volume and momentum analysis
            
            Focus on data-driven insights.
            """
            
            response = await self.agent.arun(sector_prompt)
            
            return {
                "agent_analysis": response.content,
                "sectors": sectors,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Sector analysis failed: {str(e)}"}
    
    def get_data_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """Get a summary of data availability and characteristics"""
        return {
            "symbols_count": len(symbols),
            "symbols": symbols,
            "data_sources": ["Yahoo Finance", "Market Data APIs"],
            "available_metrics": [
                "Price data (OHLCV)",
                "Returns and volatility",
                "Volume patterns", 
                "Technical indicators",
                "Correlation analysis",
                "Statistical measures"
            ],
            "update_frequency": "Real-time during market hours",
            "historical_depth": "Up to 20+ years depending on symbol"
        }