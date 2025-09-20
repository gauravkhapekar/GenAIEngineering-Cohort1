"""
Financial Modeling Team - Main Coordinator
Built with phidata for intelligent agent orchestration
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools

# Import our specialized agents
from agents.data_analyst import DataAnalystAgent
from agents.risk_assessor import RiskAssessmentAgent
from agents.portfolio_optimizer import PortfolioOptimizerAgent
from agents.forecaster import FinancialForecasterAgent
from agents.report_generator import ReportGeneratorAgent
from utils.financial_utils import FinancialCalculator
from utils.data_sources import DataSourceManager


class FinancialModelingTeam:
    """
    Orchestrates a team of specialized financial modeling agents
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the financial modeling team"""
        self.model = OpenAIChat(model=model_name)
        self.data_sources = DataSourceManager()
        self.calculator = FinancialCalculator()
        
        # Initialize specialized agents
        self.data_analyst = DataAnalystAgent(model=self.model)
        self.risk_assessor = RiskAssessmentAgent(model=self.model)
        self.portfolio_optimizer = PortfolioOptimizerAgent(model=self.model)
        self.forecaster = FinancialForecasterAgent(model=self.model)
        self.report_generator = ReportGeneratorAgent(model=self.model)
        
        # Team coordinator agent
        self.coordinator = Agent(
            name="Financial Team Coordinator",
            model=self.model,
            description="Coordinates the financial modeling team and manages workflows",
            instructions=[
                "You are the lead coordinator for a team of financial modeling specialists",
                "Delegate tasks to appropriate team members based on their expertise",
                "Synthesize results from multiple agents into cohesive insights",
                "Ensure all analysis follows best practices in financial modeling",
                "Provide clear, actionable recommendations based on team analysis"
            ],
            tools=[YFinanceTools()]
        )
        
        print("🏦 Financial Modeling Team initialized successfully!")
        print("Team members:")
        print("  📊 Data Analyst Agent")
        print("  ⚠️  Risk Assessment Agent") 
        print("  🎯 Portfolio Optimizer Agent")
        print("  🔮 Financial Forecaster Agent")
        print("  📋 Report Generator Agent")
        print("  🤝 Team Coordinator")
    
    async def analyze_portfolio(self, 
                              symbols: List[str], 
                              timeframe: str = "1Y",
                              analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis using the agent team
        
        Args:
            symbols: List of stock symbols to analyze
            timeframe: Analysis timeframe (1M, 3M, 6M, 1Y, 2Y, 5Y)
            analysis_type: Type of analysis (basic, comprehensive, risk-focused)
        
        Returns:
            Dictionary containing analysis results from all agents
        """
        print(f"\n🔍 Starting {analysis_type} portfolio analysis for: {', '.join(symbols)}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "timeframe": timeframe,
            "analysis_type": analysis_type
        }
        
        try:
            # Step 1: Data Collection and Preprocessing
            print("📊 Data Analyst collecting and processing data...")
            data_analysis = await self.data_analyst.analyze_securities(symbols, timeframe)
            results["data_analysis"] = data_analysis
            
            # Step 2: Risk Assessment
            print("⚠️  Risk Assessor evaluating portfolio risks...")
            risk_analysis = await self.risk_assessor.assess_portfolio_risk(symbols, timeframe)
            results["risk_analysis"] = risk_analysis
            
            # Step 3: Portfolio Optimization (if comprehensive analysis)
            if analysis_type in ["comprehensive", "optimization-focused"]:
                print("🎯 Portfolio Optimizer finding optimal allocations...")
                optimization = await self.portfolio_optimizer.optimize_portfolio(symbols, timeframe)
                results["optimization"] = optimization
            
            # Step 4: Forecasting (if comprehensive analysis)
            if analysis_type in ["comprehensive", "forecast-focused"]:
                print("🔮 Financial Forecaster generating predictions...")
                forecasts = await self.forecaster.generate_forecasts(symbols, timeframe)
                results["forecasts"] = forecasts
            
            # Step 5: Coordinator synthesis
            print("🤝 Team Coordinator synthesizing results...")
            coordinator_response = await self.coordinator.arun(
                f"Analyze the portfolio {symbols} based on team member findings. "
                f"Data Analysis: {data_analysis}. "
                f"Risk Analysis: {risk_analysis}. "
                f"Provide key insights and recommendations."
            )
            results["coordinator_insights"] = coordinator_response.content
            
            print("✅ Portfolio analysis completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Error during portfolio analysis: {str(e)}")
            results["error"] = str(e)
            return results
    
    async def assess_portfolio_risk(self, 
                                   symbols: List[str], 
                                   timeframe: str = "1Y") -> Dict[str, Any]:
        """
        Focused risk assessment of a portfolio
        """
        print(f"\n⚠️  Starting risk assessment for: {', '.join(symbols)}")
        
        try:
            # Get detailed risk analysis
            risk_analysis = await self.risk_assessor.comprehensive_risk_analysis(symbols, timeframe)
            
            # Get coordinator's risk interpretation
            risk_insights = await self.coordinator.arun(
                f"Interpret this risk analysis for portfolio {symbols}: {risk_analysis}. "
                "Provide clear risk warnings and mitigation strategies."
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "timeframe": timeframe,
                "detailed_risk_analysis": risk_analysis,
                "risk_insights": risk_insights.content
            }
            
        except Exception as e:
            print(f"❌ Error during risk assessment: {str(e)}")
            return {"error": str(e)}
    
    async def generate_comprehensive_report(self, 
                                          symbols: List[str],
                                          timeframe: str = "1Y",
                                          output_format: str = "json") -> Dict[str, Any]:
        """
        Generate a comprehensive financial report
        """
        print(f"\n📋 Generating comprehensive report for: {', '.join(symbols)}")
        
        try:
            # Get full analysis
            analysis = await self.analyze_portfolio(symbols, timeframe, "comprehensive")
            
            # Generate formatted report
            report = await self.report_generator.create_comprehensive_report(
                analysis, output_format
            )
            
            print("✅ Comprehensive report generated!")
            return report
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
            return {"error": str(e)}
    
    def get_team_status(self) -> Dict[str, str]:
        """Get status of all team members"""
        return {
            "data_analyst": "Ready for data collection and analysis",
            "risk_assessor": "Ready for risk evaluation",
            "portfolio_optimizer": "Ready for optimization tasks", 
            "forecaster": "Ready for predictive modeling",
            "report_generator": "Ready for report creation",
            "coordinator": "Ready to orchestrate team workflows"
        }
    
    async def quick_market_insight(self, symbols: List[str]) -> str:
        """Get quick market insights for given symbols"""
        insight = await self.coordinator.arun(
            f"Provide quick market insights for {symbols}. "
            "Include current price trends, key metrics, and brief outlook."
        )
        return insight.content


async def main():
    """Main function to demonstrate the financial modeling team"""
    
    # Initialize the team
    team = FinancialModelingTeam()
    
    # Sample portfolio for demonstration
    sample_portfolio = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    print("\n" + "="*60)
    print("🏦 FINANCIAL MODELING TEAM DEMONSTRATION")
    print("="*60)
    
    # Check team status
    print("\n📋 Team Status:")
    status = team.get_team_status()
    for agent, status_msg in status.items():
        print(f"  {agent}: {status_msg}")
    
    # Demonstrate quick market insight
    print(f"\n💡 Quick Market Insight for {sample_portfolio}:")
    quick_insight = await team.quick_market_insight(sample_portfolio)
    print(quick_insight)
    
    # Demonstrate comprehensive analysis
    print(f"\n🔍 Running comprehensive analysis...")
    analysis_results = await team.analyze_portfolio(
        symbols=sample_portfolio,
        timeframe="6M",
        analysis_type="comprehensive"
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/outputs/portfolio_analysis_{timestamp}.json"
    
    try:
        import os
        os.makedirs("data/outputs", exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    print("\n✅ Financial Modeling Team demonstration completed!")
    print("🎯 Next steps: Customize agents, add data sources, or run specific analyses")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())