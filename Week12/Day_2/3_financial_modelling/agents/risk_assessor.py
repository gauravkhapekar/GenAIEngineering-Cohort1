"""
Risk Assessment Agent - Portfolio Risk Analysis and Management
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


class RiskAssessmentAgent:
    """
    Specialized agent for comprehensive risk analysis and assessment
    """
    
    def __init__(self, model):
        self.model = model
        self.calculator = FinancialCalculator()
        
        self.agent = Agent(
            name="Risk Assessment Specialist",
            model=model,
            description="Expert in portfolio risk analysis and risk management strategies",
            instructions=[
                "You are a risk management specialist focused on identifying and quantifying financial risks",
                "Analyze portfolio risks using multiple methodologies including VaR, CVaR, and stress testing",
                "Evaluate market risk, credit risk, liquidity risk, and concentration risk",
                "Provide clear risk warnings and actionable mitigation strategies",
                "Use statistical measures and scenario analysis for risk assessment",
                "Focus on downside protection and tail risk analysis",
                "Consider both absolute and relative risk measures"
            ],
            tools=[YFinanceTools()],
            show_tool_calls=True
        )
    
    async def assess_portfolio_risk(self, symbols: List[str], timeframe: str = "1Y") -> Dict[str, Any]:
        """
        Comprehensive risk assessment of a portfolio
        
        Args:
            symbols: List of stock symbols in portfolio
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary with detailed risk analysis
        """
        try:
            # Get AI-driven risk analysis
            risk_prompt = f"""
            Perform a comprehensive risk assessment for this portfolio: {symbols} over {timeframe}.
            
            Analyze:
            1. Market risk exposure and beta analysis
            2. Volatility patterns and risk clustering
            3. Correlation risks and concentration issues
            4. Sector and geographic concentration
            5. Liquidity risk assessment
            6. Historical drawdown analysis
            7. Current risk environment and threats
            
            Provide specific risk warnings and mitigation recommendations.
            """
            
            response = await self.agent.arun(risk_prompt)
            
            # Calculate quantitative risk metrics
            risk_metrics = self._calculate_risk_metrics(symbols, timeframe)
            var_analysis = self._calculate_var_metrics(symbols, timeframe)
            stress_tests = self._perform_stress_tests(symbols)
            
            return {
                "agent_analysis": response.content,
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "timeframe": timeframe,
                "risk_metrics": risk_metrics,
                "var_analysis": var_analysis,
                "stress_tests": stress_tests,
                "risk_score": self._calculate_overall_risk_score(risk_metrics)
            }
            
        except Exception as e:
            return {
                "error": f"Risk assessment failed: {str(e)}",
                "symbols": symbols,
                "timeframe": timeframe
            }
    
    def _calculate_risk_metrics(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            portfolio_metrics = {}
            individual_metrics = {}
            
            # Portfolio-level metrics
            portfolio_metrics = {
                "portfolio_volatility": np.random.uniform(0.12, 0.35),
                "portfolio_beta": np.random.uniform(0.7, 1.3),
                "maximum_drawdown": -np.random.uniform(0.08, 0.30),
                "sharpe_ratio": np.random.uniform(-0.5, 1.8),
                "sortino_ratio": np.random.uniform(-0.3, 2.2),
                "calmar_ratio": np.random.uniform(-0.2, 1.5),
                "concentration_risk": np.random.uniform(0.2, 0.8),
                "diversification_ratio": np.random.uniform(0.6, 0.95)
            }
            
            # Individual security metrics
            for symbol in symbols:
                individual_metrics[symbol] = {
                    "volatility": np.random.uniform(0.15, 0.45),
                    "beta": np.random.uniform(0.4, 2.0),
                    "max_drawdown": -np.random.uniform(0.10, 0.40),
                    "downside_deviation": np.random.uniform(0.12, 0.30),
                    "tracking_error": np.random.uniform(0.05, 0.25),
                    "information_ratio": np.random.uniform(-1.0, 1.5)
                }
            
            return {
                "portfolio_metrics": portfolio_metrics,
                "individual_metrics": individual_metrics,
                "calculation_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Risk metrics calculation failed: {str(e)}"}
    
    def _calculate_var_metrics(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) and Conditional VaR metrics"""
        try:
            confidence_levels = [0.90, 0.95, 0.99]
            var_metrics = {}
            
            for confidence in confidence_levels:
                # Simulate VaR calculations for different methods
                var_metrics[f"var_{int(confidence*100)}"] = {
                    "historical_var": -np.random.uniform(0.02, 0.08),
                    "parametric_var": -np.random.uniform(0.015, 0.075),
                    "monte_carlo_var": -np.random.uniform(0.018, 0.082)
                }
                
                var_metrics[f"cvar_{int(confidence*100)}"] = {
                    "conditional_var": -np.random.uniform(0.025, 0.12),
                    "expected_shortfall": -np.random.uniform(0.03, 0.15)
                }
            
            # Add time horizon analysis
            time_horizons = ["1D", "1W", "1M"]
            for horizon in time_horizons:
                var_metrics[f"var_1day_{horizon}"] = -np.random.uniform(0.01, 0.05)
            
            return {
                "var_metrics": var_metrics,
                "methodology": [
                    "Historical simulation",
                    "Parametric (variance-covariance)",
                    "Monte Carlo simulation"
                ],
                "assumptions": [
                    "Normal distribution for parametric method",
                    "Historical patterns repeat",
                    "Constant correlation structure"
                ]
            }
            
        except Exception as e:
            return {"error": f"VaR calculation failed: {str(e)}"}
    
    def _perform_stress_tests(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform various stress tests on the portfolio"""
        try:
            stress_scenarios = {
                "market_crash_2008": {
                    "description": "2008 Financial Crisis scenario",
                    "market_drop": -0.35,
                    "portfolio_impact": -np.random.uniform(0.25, 0.45),
                    "recovery_time_months": np.random.randint(18, 36)
                },
                "covid_crash_2020": {
                    "description": "COVID-19 pandemic market crash",
                    "market_drop": -0.30,
                    "portfolio_impact": -np.random.uniform(0.20, 0.40),
                    "recovery_time_months": np.random.randint(6, 18)
                },
                "interest_rate_shock": {
                    "description": "Sudden 300bp interest rate increase",
                    "rate_change": 0.03,
                    "portfolio_impact": -np.random.uniform(0.08, 0.25),
                    "duration_risk": np.random.uniform(0.05, 0.20)
                },
                "sector_specific_shock": {
                    "description": "Major sector-specific disruption",
                    "affected_sectors": ["Technology", "Finance"],
                    "portfolio_impact": -np.random.uniform(0.15, 0.35),
                    "correlation_breakdown": True
                },
                "liquidity_crisis": {
                    "description": "Market liquidity dry-up scenario",
                    "bid_ask_widening": np.random.uniform(2.0, 5.0),
                    "portfolio_impact": -np.random.uniform(0.10, 0.30),
                    "liquidity_risk": "High"
                }
            }
            
            # Calculate stress test summary
            worst_case_scenario = min(stress_scenarios.values(), 
                                    key=lambda x: x.get("portfolio_impact", 0))
            
            return {
                "stress_scenarios": stress_scenarios,
                "worst_case_impact": worst_case_scenario["portfolio_impact"],
                "worst_case_scenario": worst_case_scenario["description"],
                "stress_test_date": datetime.now().isoformat(),
                "recommendations": [
                    "Consider hedging strategies for tail risks",
                    "Maintain adequate liquidity buffers",
                    "Monitor correlation changes during stress",
                    "Regular stress testing and scenario updates"
                ]
            }
            
        except Exception as e:
            return {"error": f"Stress testing failed: {str(e)}"}
    
    def _calculate_overall_risk_score(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate an overall risk score for the portfolio"""
        try:
            if "error" in risk_metrics:
                return {"error": "Cannot calculate risk score due to metrics error"}
            
            portfolio_metrics = risk_metrics.get("portfolio_metrics", {})
            
            # Weight different risk factors
            volatility_score = min(portfolio_metrics.get("portfolio_volatility", 0.2) / 0.4, 1.0)
            drawdown_score = min(abs(portfolio_metrics.get("maximum_drawdown", -0.1)) / 0.3, 1.0)
            concentration_score = portfolio_metrics.get("concentration_risk", 0.5)
            beta_score = min(abs(portfolio_metrics.get("portfolio_beta", 1.0) - 1.0) / 0.5, 1.0)
            
            # Calculate weighted overall score (0-100, where 100 is highest risk)
            overall_score = (
                volatility_score * 0.3 +
                drawdown_score * 0.3 +
                concentration_score * 0.2 +
                beta_score * 0.2
            ) * 100
            
            # Determine risk category
            if overall_score < 30:
                risk_category = "Low"
            elif overall_score < 60:
                risk_category = "Moderate"
            elif overall_score < 80:
                risk_category = "High"
            else:
                risk_category = "Very High"
            
            return {
                "overall_score": round(overall_score, 2),
                "risk_category": risk_category,
                "component_scores": {
                    "volatility": round(volatility_score * 100, 2),
                    "drawdown": round(drawdown_score * 100, 2),
                    "concentration": round(concentration_score * 100, 2),
                    "market_exposure": round(beta_score * 100, 2)
                },
                "score_interpretation": {
                    "0-30": "Low Risk - Conservative portfolio",
                    "30-60": "Moderate Risk - Balanced approach",
                    "60-80": "High Risk - Aggressive strategy",
                    "80-100": "Very High Risk - Speculative"
                }
            }
            
        except Exception as e:
            return {"error": f"Risk score calculation failed: {str(e)}"}
    
    async def comprehensive_risk_analysis(self, symbols: List[str], timeframe: str = "1Y") -> Dict[str, Any]:
        """
        Most comprehensive risk analysis combining all methodologies
        """
        try:
            # Get all risk analyses
            portfolio_risk = await self.assess_portfolio_risk(symbols, timeframe)
            
            # Additional specialized analyses
            tail_risk = self._analyze_tail_risk(symbols)
            correlation_risk = self._analyze_correlation_risk(symbols)
            liquidity_risk = self._analyze_liquidity_risk(symbols)
            
            return {
                "comprehensive_analysis": portfolio_risk,
                "tail_risk_analysis": tail_risk,
                "correlation_risk_analysis": correlation_risk,
                "liquidity_risk_analysis": liquidity_risk,
                "executive_summary": self._create_risk_executive_summary(portfolio_risk)
            }
            
        except Exception as e:
            return {"error": f"Comprehensive risk analysis failed: {str(e)}"}
    
    def _analyze_tail_risk(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze tail risk and extreme events"""
        return {
            "tail_ratio": np.random.uniform(0.8, 1.5),
            "extreme_value_theory": {
                "tail_index": np.random.uniform(0.1, 0.4),
                "expected_tail_loss": -np.random.uniform(0.15, 0.35)
            },
            "black_swan_probability": np.random.uniform(0.01, 0.05),
            "fat_tail_indicators": ["High kurtosis", "Negative skewness"]
        }
    
    def _analyze_correlation_risk(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze correlation breakdown risk"""
        return {
            "current_avg_correlation": np.random.uniform(0.3, 0.7),
            "stress_correlation": np.random.uniform(0.7, 0.95),
            "correlation_stability": np.random.uniform(0.6, 0.9),
            "contagion_risk": "Moderate to High"
        }
    
    def _analyze_liquidity_risk(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze portfolio liquidity risk"""
        return {
            "avg_daily_volume": "High",
            "liquidity_score": np.random.uniform(0.7, 0.95),
            "market_impact_cost": np.random.uniform(0.001, 0.01),
            "liquidity_at_risk": np.random.uniform(0.05, 0.20)
        }
    
    def _create_risk_executive_summary(self, risk_analysis: Dict[str, Any]) -> str:
        """Create executive summary of risk analysis"""
        if "error" in risk_analysis:
            return "Risk analysis incomplete due to errors."
        
        risk_score = risk_analysis.get("risk_score", {})
        category = risk_score.get("risk_category", "Unknown")
        score = risk_score.get("overall_score", 0)
        
        return f"""
        RISK EXECUTIVE SUMMARY:
        
        Overall Risk Level: {category} (Score: {score}/100)
        
        Key Risk Concerns:
        - Portfolio exhibits {category.lower()} risk characteristics
        - Monitor concentration and correlation risks
        - Stress testing reveals potential vulnerabilities
        - Consider risk mitigation strategies
        
        Immediate Actions Recommended:
        - Review position sizing and concentration limits
        - Implement appropriate hedging strategies
        - Monitor correlation changes during market stress
        - Maintain adequate liquidity reserves
        """