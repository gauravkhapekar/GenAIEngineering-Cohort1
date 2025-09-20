"""
Report Generator Agent - Financial Report Creation and Visualization
Part of the Financial Modeling Team
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from utils.financial_utils import FinancialCalculator


class ReportGeneratorAgent:
    """
    Specialized agent for creating comprehensive financial reports and visualizations
    """
    
    def __init__(self, model):
        self.model = model
        self.calculator = FinancialCalculator()
        
        self.agent = Agent(
            name="Financial Report Generator",
            model=model,
            description="Expert in creating professional financial reports and analysis documents",
            instructions=[
                "You are a financial reporting specialist focused on clear, professional communication",
                "Create comprehensive reports that synthesize complex financial analysis",
                "Use clear executive summaries and actionable insights",
                "Structure reports for different audiences: executives, analysts, investors",
                "Include both quantitative analysis and qualitative insights",
                "Highlight key risks, opportunities, and recommendations",
                "Ensure reports are factual, balanced, and professionally formatted"
            ],
            tools=[YFinanceTools()],
            show_tool_calls=True
        )
    
    async def create_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                        output_format: str = "json") -> Dict[str, Any]:
        """
        Create a comprehensive financial report from analysis results
        
        Args:
            analysis_results: Combined results from all team agents
            output_format: Format for the report (json, html, markdown)
            
        Returns:
            Formatted financial report
        """
        try:
            # Generate executive summary using AI
            summary_prompt = f"""
            Create an executive summary for this financial analysis: {json.dumps(analysis_results, default=str)[:2000]}...
            
            The summary should include:
            1. Key findings and insights
            2. Portfolio performance assessment
            3. Risk evaluation summary
            4. Investment recommendations
            5. Action items and next steps
            
            Keep it concise, professional, and actionable for senior management.
            """
            
            summary_response = await self.agent.arun(summary_prompt)
            
            # Create structured report
            report = self._structure_comprehensive_report(analysis_results, summary_response.content)
            
            # Format according to requested output type
            if output_format == "html":
                formatted_report = self._format_html_report(report)
            elif output_format == "markdown":
                formatted_report = self._format_markdown_report(report)
            else:
                formatted_report = report  # JSON format
            
            return {
                "report": formatted_report,
                "format": output_format,
                "generation_timestamp": datetime.now().isoformat(),
                "report_metadata": self._generate_report_metadata(analysis_results)
            }
            
        except Exception as e:
            return {
                "error": f"Report generation failed: {str(e)}",
                "analysis_results": analysis_results
            }
    
    def _structure_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                      executive_summary: str) -> Dict[str, Any]:
        """Structure the comprehensive report with all sections"""
        
        symbols = analysis_results.get("symbols", [])
        timestamp = analysis_results.get("timestamp", datetime.now().isoformat())
        
        report = {
            "report_header": {
                "title": "Comprehensive Financial Analysis Report",
                "portfolio_symbols": symbols,
                "analysis_date": timestamp,
                "generated_by": "Financial Modeling Team",
                "report_id": f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            },
            
            "executive_summary": {
                "ai_generated_summary": executive_summary,
                "key_metrics": self._extract_key_metrics(analysis_results),
                "investment_recommendation": self._generate_investment_recommendation(analysis_results),
                "risk_assessment": self._summarize_risk_assessment(analysis_results)
            },
            
            "portfolio_overview": self._create_portfolio_overview(analysis_results),
            "performance_analysis": self._create_performance_analysis(analysis_results),
            "risk_analysis": self._create_risk_analysis_section(analysis_results),
            "optimization_results": self._create_optimization_section(analysis_results),
            "forecasting_insights": self._create_forecasting_section(analysis_results),
            "recommendations": self._create_recommendations_section(analysis_results),
            "appendices": self._create_appendices(analysis_results)
        }
        
        return report
    
    def _extract_key_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key portfolio metrics for executive summary"""
        key_metrics = {}
        
        # Extract from data analysis
        data_analysis = analysis_results.get("data_analysis", {})
        if "calculated_metrics" in data_analysis:
            metrics = data_analysis["calculated_metrics"]
            if metrics:
                symbol = list(metrics.keys())[0]  # Take first symbol as representative
                key_metrics["representative_sharpe_ratio"] = metrics[symbol].get("sharpe_ratio", "N/A")
                key_metrics["representative_volatility"] = metrics[symbol].get("annualized_volatility", "N/A")
        
        # Extract from risk analysis
        risk_analysis = analysis_results.get("risk_analysis", {})
        if "risk_score" in risk_analysis:
            risk_score = risk_analysis["risk_score"]
            key_metrics["overall_risk_score"] = risk_score.get("overall_score", "N/A")
            key_metrics["risk_category"] = risk_score.get("risk_category", "N/A")
        
        # Extract from optimization
        optimization = analysis_results.get("optimization", {})
        if "performance_projections" in optimization:
            projections = optimization["performance_projections"]
            key_metrics["expected_annual_return"] = projections.get("expected_annual_return", "N/A")
            key_metrics["expected_annual_volatility"] = projections.get("expected_annual_volatility", "N/A")
        
        return key_metrics
    
    def _generate_investment_recommendation(self, analysis_results: Dict[str, Any]) -> str:
        """Generate overall investment recommendation"""
        
        # Extract risk score
        risk_analysis = analysis_results.get("risk_analysis", {})
        risk_score = risk_analysis.get("risk_score", {})
        risk_category = risk_score.get("risk_category", "Unknown")
        
        # Extract optimization results
        optimization = analysis_results.get("optimization", {})
        
        # Generate recommendation based on analysis
        if risk_category == "Low":
            return "CONSERVATIVE: Portfolio exhibits low risk characteristics. Suitable for risk-averse investors seeking capital preservation."
        elif risk_category == "Moderate":
            return "BALANCED: Portfolio shows moderate risk-return profile. Appropriate for investors with balanced risk tolerance."
        elif risk_category == "High":
            return "AGGRESSIVE: High-risk portfolio requiring active monitoring. Suitable for risk-tolerant investors with growth objectives."
        else:
            return "PROCEED WITH CAUTION: Risk assessment incomplete or concerning. Recommend further analysis before investment decisions."
    
    def _summarize_risk_assessment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize key risk findings"""
        risk_analysis = analysis_results.get("risk_analysis", {})
        
        if not risk_analysis:
            return {"status": "Risk analysis not available"}
        
        risk_metrics = risk_analysis.get("risk_metrics", {})
        portfolio_metrics = risk_metrics.get("portfolio_metrics", {}) if risk_metrics else {}
        
        return {
            "maximum_drawdown": portfolio_metrics.get("maximum_drawdown", "N/A"),
            "portfolio_volatility": portfolio_metrics.get("portfolio_volatility", "N/A"),
            "concentration_risk": portfolio_metrics.get("concentration_risk", "N/A"),
            "diversification_ratio": portfolio_metrics.get("diversification_ratio", "N/A"),
            "key_concerns": [
                "Monitor concentration levels",
                "Watch correlation changes during stress",
                "Consider hedging for tail risks"
            ]
        }
    
    def _create_portfolio_overview(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create portfolio overview section"""
        symbols = analysis_results.get("symbols", [])
        
        return {
            "portfolio_composition": {
                "number_of_securities": len(symbols),
                "securities": symbols,
                "analysis_timeframe": analysis_results.get("timeframe", "N/A")
            },
            "data_quality": analysis_results.get("data_analysis", {}).get("data_quality", {}),
            "correlation_summary": analysis_results.get("data_analysis", {}).get("correlation_matrix", {})
        }
    
    def _create_performance_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance analysis section"""
        data_analysis = analysis_results.get("data_analysis", {})
        calculated_metrics = data_analysis.get("calculated_metrics", {})
        
        if not calculated_metrics:
            return {"status": "Performance metrics not available"}
        
        # Aggregate performance metrics across securities
        all_sharpe_ratios = [metrics.get("sharpe_ratio", 0) for metrics in calculated_metrics.values()]
        all_volatilities = [metrics.get("annualized_volatility", 0) for metrics in calculated_metrics.values()]
        
        return {
            "individual_performance": calculated_metrics,
            "portfolio_summary": {
                "average_sharpe_ratio": np.mean(all_sharpe_ratios) if all_sharpe_ratios else "N/A",
                "average_volatility": np.mean(all_volatilities) if all_volatilities else "N/A",
                "best_performer": max(calculated_metrics.items(), key=lambda x: x[1].get("sharpe_ratio", 0))[0] if calculated_metrics else "N/A",
                "highest_volatility": max(calculated_metrics.items(), key=lambda x: x[1].get("annualized_volatility", 0))[0] if calculated_metrics else "N/A"
            }
        }
    
    def _create_risk_analysis_section(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed risk analysis section"""
        risk_analysis = analysis_results.get("risk_analysis", {})
        
        if not risk_analysis:
            return {"status": "Risk analysis not available"}
        
        return {
            "risk_metrics": risk_analysis.get("risk_metrics", {}),
            "var_analysis": risk_analysis.get("var_analysis", {}),
            "stress_tests": risk_analysis.get("stress_tests", {}),
            "overall_risk_assessment": risk_analysis.get("risk_score", {}),
            "risk_recommendations": [
                "Regular monitoring of risk metrics",
                "Stress testing under various scenarios",
                "Consider risk mitigation strategies",
                "Monitor correlation changes"
            ]
        }
    
    def _create_optimization_section(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create portfolio optimization section"""
        optimization = analysis_results.get("optimization", {})
        
        if not optimization:
            return {"status": "Optimization results not available"}
        
        return {
            "efficient_frontier": optimization.get("efficient_frontier", {}),
            "optimal_allocations": optimization.get("optimal_allocations", {}),
            "performance_projections": optimization.get("performance_projections", {}),
            "implementation_notes": optimization.get("implementation_notes", []),
            "optimization_insights": [
                "Diversification benefits identified",
                "Risk-return tradeoffs analyzed",
                "Implementation considerations provided"
            ]
        }
    
    def _create_forecasting_section(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create forecasting insights section"""
        forecasts = analysis_results.get("forecasts", {})
        
        if not forecasts:
            return {"status": "Forecasting results not available"}
        
        return {
            "time_series_forecasts": forecasts.get("time_series_forecasts", {}),
            "technical_forecasts": forecasts.get("technical_forecasts", {}),
            "fundamental_forecasts": forecasts.get("fundamental_forecasts", {}),
            "monte_carlo_simulations": forecasts.get("monte_carlo_simulations", {}),
            "ensemble_forecast": forecasts.get("ensemble_forecast", {}),
            "forecast_limitations": [
                "Forecasts are probabilistic, not deterministic",
                "Model uncertainty exists",
                "External shocks cannot be predicted",
                "Regular model updates recommended"
            ]
        }
    
    def _create_recommendations_section(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create actionable recommendations section"""
        
        symbols = analysis_results.get("symbols", [])
        risk_analysis = analysis_results.get("risk_analysis", {})
        
        # Generate specific recommendations based on analysis
        recommendations = {
            "immediate_actions": [
                "Review current position sizes for concentration risk",
                "Monitor key risk metrics on daily basis",
                "Assess liquidity requirements"
            ],
            "strategic_considerations": [
                "Consider rebalancing based on optimization results",
                "Evaluate hedging strategies for tail risks",
                "Review investment policy statement"
            ],
            "monitoring_requirements": [
                "Daily risk monitoring",
                "Weekly performance review",
                "Monthly rebalancing assessment",
                "Quarterly strategy review"
            ]
        }
        
        # Add specific recommendations based on risk level
        risk_score = risk_analysis.get("risk_score", {})
        risk_category = risk_score.get("risk_category", "Unknown")
        
        if risk_category == "High" or risk_category == "Very High":
            recommendations["immediate_actions"].append("Consider reducing position sizes")
            recommendations["immediate_actions"].append("Implement stop-loss levels")
        
        return recommendations
    
    def _create_appendices(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create appendices with detailed data and methodologies"""
        return {
            "methodology": {
                "data_sources": ["Yahoo Finance", "Market Data APIs"],
                "risk_models": ["VaR", "Stress Testing", "Monte Carlo"],
                "optimization_methods": ["Mean-Variance", "Risk Parity"],
                "forecasting_models": ["ARIMA", "GARCH", "Technical Analysis"]
            },
            "disclaimers": [
                "This analysis is for informational purposes only",
                "Past performance does not guarantee future results",
                "All investments carry risk of loss",
                "Consult with financial advisor before making decisions"
            ],
            "data_sources": {
                "market_data": "Yahoo Finance API",
                "fundamental_data": "Simulated for demonstration",
                "economic_data": "Various public sources"
            },
            "contact_information": {
                "generated_by": "Financial Modeling Team",
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "version": "1.0"
            }
        }
    
    def _format_html_report(self, report: Dict[str, Any]) -> str:
        """Format report as HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report['report_header']['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .metric {{ background: #f8f9fa; padding: 10px; margin: 5px; border-radius: 5px; }}
                .risk-high {{ color: #e74c3c; }}
                .risk-moderate {{ color: #f39c12; }}
                .risk-low {{ color: #27ae60; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{report['report_header']['title']}</h1>
            <div class="report-info">
                <p><strong>Portfolio:</strong> {', '.join(report['report_header']['portfolio_symbols'])}</p>
                <p><strong>Analysis Date:</strong> {report['report_header']['analysis_date']}</p>
                <p><strong>Report ID:</strong> {report['report_header']['report_id']}</p>
            </div>
            
            <h2>Executive Summary</h2>
            <div class="executive-summary">
                {report['executive_summary']['ai_generated_summary']}
            </div>
            
            <h2>Key Metrics</h2>
            <div class="key-metrics">
                {self._format_metrics_html(report['executive_summary']['key_metrics'])}
            </div>
            
            <h2>Investment Recommendation</h2>
            <p>{report['executive_summary']['investment_recommendation']}</p>
            
            <div class="disclaimer">
                <h3>Important Disclaimers</h3>
                <ul>
                    {"".join(f"<li>{disclaimer}</li>" for disclaimer in report['appendices']['disclaimers'])}
                </ul>
            </div>
        </body>
        </html>
        """
        return html_content
    
    def _format_metrics_html(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML"""
        html = ""
        for key, value in metrics.items():
            html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        return html
    
    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format report as Markdown"""
        markdown_content = f"""
# {report['report_header']['title']}

**Portfolio:** {', '.join(report['report_header']['portfolio_symbols'])}
**Analysis Date:** {report['report_header']['analysis_date']}
**Report ID:** {report['report_header']['report_id']}

## Executive Summary

{report['executive_summary']['ai_generated_summary']}

### Key Metrics
{self._format_metrics_markdown(report['executive_summary']['key_metrics'])}

### Investment Recommendation
{report['executive_summary']['investment_recommendation']}

## Risk Assessment Summary
- **Maximum Drawdown:** {report['executive_summary']['risk_assessment'].get('maximum_drawdown', 'N/A')}
- **Portfolio Volatility:** {report['executive_summary']['risk_assessment'].get('portfolio_volatility', 'N/A')}
- **Concentration Risk:** {report['executive_summary']['risk_assessment'].get('concentration_risk', 'N/A')}

## Recommendations

### Immediate Actions
{chr(10).join(f"- {action}" for action in report['recommendations']['immediate_actions'])}

### Strategic Considerations  
{chr(10).join(f"- {consideration}" for consideration in report['recommendations']['strategic_considerations'])}

## Disclaimers
{chr(10).join(f"- {disclaimer}" for disclaimer in report['appendices']['disclaimers'])}

---
*Generated by Financial Modeling Team on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
        """
        return markdown_content
    
    def _format_metrics_markdown(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as Markdown"""
        markdown = ""
        for key, value in metrics.items():
            markdown += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        return markdown
    
    def _generate_report_metadata(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for the report"""
        return {
            "report_version": "1.0",
            "analysis_components": list(analysis_results.keys()),
            "generation_time": datetime.now().isoformat(),
            "data_quality_score": self._assess_overall_data_quality(analysis_results),
            "completeness_score": self._assess_analysis_completeness(analysis_results)
        }
    
    def _assess_overall_data_quality(self, analysis_results: Dict[str, Any]) -> float:
        """Assess overall data quality score"""
        data_analysis = analysis_results.get("data_analysis", {})
        data_quality = data_analysis.get("data_quality", {})
        
        if not data_quality:
            return 0.5  # Default score when no data quality info
        
        # Calculate average quality score across all symbols
        quality_scores = []
        for symbol_quality in data_quality.values():
            if isinstance(symbol_quality, dict) and "quality_score" in symbol_quality:
                quality_scores.append(symbol_quality["quality_score"])
        
        return np.mean(quality_scores) if quality_scores else 0.5
    
    def _assess_analysis_completeness(self, analysis_results: Dict[str, Any]) -> float:
        """Assess completeness of the analysis"""
        expected_components = [
            "data_analysis", "risk_analysis", "optimization", 
            "forecasts", "coordinator_insights"
        ]
        
        present_components = sum(1 for component in expected_components 
                               if component in analysis_results and analysis_results[component])
        
        return present_components / len(expected_components)
    
    async def create_risk_report(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a focused risk report"""
        try:
            risk_prompt = f"""
            Create a focused risk report based on this analysis: {json.dumps(risk_analysis, default=str)[:1500]}...
            
            Focus on:
            1. Key risk exposures and concerns
            2. Stress test results and implications  
            3. Risk mitigation recommendations
            4. Monitoring requirements
            
            Make it actionable for risk managers.
            """
            
            response = await self.agent.arun(risk_prompt)
            
            return {
                "risk_report": {
                    "executive_summary": response.content,
                    "key_risks": self._extract_key_risks(risk_analysis),
                    "recommendations": self._generate_risk_recommendations(risk_analysis),
                    "monitoring_dashboard": self._create_risk_monitoring_dashboard(risk_analysis)
                },
                "generation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Risk report generation failed: {str(e)}"}
    
    def _extract_key_risks(self, risk_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key risks from analysis"""
        risks = []
        
        risk_score = risk_analysis.get("risk_score", {})
        if risk_score:
            overall_score = risk_score.get("overall_score", 0)
            if overall_score > 70:
                risks.append({
                    "risk_type": "Overall Portfolio Risk",
                    "severity": "High",
                    "description": f"Portfolio risk score of {overall_score} indicates elevated risk levels"
                })
        
        stress_tests = risk_analysis.get("stress_tests", {})
        if stress_tests and "stress_scenarios" in stress_tests:
            for scenario, details in stress_tests["stress_scenarios"].items():
                impact = details.get("portfolio_impact", 0)
                if impact < -0.20:  # More than 20% loss
                    risks.append({
                        "risk_type": "Stress Test Risk",
                        "severity": "High" if impact < -0.30 else "Moderate",
                        "description": f"{scenario}: Potential loss of {impact:.1%}"
                    })
        
        return risks
    
    def _generate_risk_recommendations(self, risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate risk-specific recommendations"""
        recommendations = [
            "Implement daily risk monitoring processes",
            "Set position size limits to control concentration",
            "Consider hedging strategies for tail risks",
            "Regular stress testing and scenario analysis"
        ]
        
        risk_score = risk_analysis.get("risk_score", {})
        risk_category = risk_score.get("risk_category", "")
        
        if risk_category in ["High", "Very High"]:
            recommendations.extend([
                "Reduce overall portfolio risk exposure",
                "Implement stop-loss mechanisms",
                "Increase monitoring frequency"
            ])
        
        return recommendations
    
    def _create_risk_monitoring_dashboard(self, risk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create risk monitoring dashboard structure"""
        return {
            "daily_metrics": [
                "Portfolio VaR (95%)",
                "Maximum position size",
                "Correlation changes",
                "Volatility regime"
            ],
            "weekly_metrics": [
                "Stress test results",
                "Risk attribution analysis",
                "Concentration measures"
            ],
            "monthly_metrics": [
                "Risk model validation",
                "Scenario analysis update",
                "Risk limit review"
            ],
            "alert_thresholds": {
                "var_breach": "VaR exceeded by 20%",
                "concentration_limit": "Single position > 25%",
                "correlation_spike": "Average correlation > 0.8"
            }
        }