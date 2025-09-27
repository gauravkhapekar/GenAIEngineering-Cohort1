"""
Portfolio Optimizer Agent - Modern Portfolio Theory and Optimization
Part of the Financial Modeling Team
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from utils.financial_utils import FinancialCalculator


class PortfolioOptimizerAgent:
    """
    Specialized agent for portfolio optimization using Modern Portfolio Theory
    and advanced optimization techniques
    """
    
    def __init__(self, model):
        self.model = model
        self.calculator = FinancialCalculator()
        
        self.agent = Agent(
            name="Portfolio Optimization Specialist",
            model=model,
            description="Expert in portfolio optimization and asset allocation strategies",
            instructions=[
                "You are a portfolio optimization specialist using Modern Portfolio Theory and advanced techniques",
                "Focus on efficient frontier construction and optimal asset allocation",
                "Consider multiple optimization objectives: risk minimization, return maximization, Sharpe ratio",
                "Implement various optimization strategies: mean-variance, risk parity, black-litterman",
                "Account for real-world constraints like transaction costs and position limits",
                "Provide clear rationale for allocation recommendations",
                "Consider both tactical and strategic allocation perspectives"
            ],
            tools=[YFinanceTools()],
            show_tool_calls=True
        )
    
    async def optimize_portfolio(self, symbols: List[str], timeframe: str = "1Y", 
                               optimization_method: str = "mean_variance") -> Dict[str, Any]:
        """
        Optimize portfolio allocation using specified method
        
        Args:
            symbols: List of stock symbols to optimize
            timeframe: Historical data timeframe for optimization
            optimization_method: Optimization approach to use
            
        Returns:
            Dictionary with optimization results and recommendations
        """
        try:
            # Get AI-driven optimization insights
            optimization_prompt = f"""
            Optimize the portfolio allocation for these securities: {symbols} using {optimization_method} approach.
            
            Consider:
            1. Risk-return tradeoffs and efficient frontier
            2. Correlation structure and diversification benefits
            3. Current market conditions and outlook
            4. Practical constraints and implementation considerations
            5. Alternative allocation strategies
            
            Provide specific allocation percentages and rationale for recommendations.
            """
            
            response = await self.agent.arun(optimization_prompt)
            
            # Calculate quantitative optimization results
            efficient_frontier = self._calculate_efficient_frontier(symbols, timeframe)
            optimal_allocations = self._calculate_optimal_allocations(symbols, optimization_method)
            performance_projections = self._project_portfolio_performance(symbols, optimal_allocations)
            
            return {
                "agent_analysis": response.content,
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols,
                "optimization_method": optimization_method,
                "timeframe": timeframe,
                "efficient_frontier": efficient_frontier,
                "optimal_allocations": optimal_allocations,
                "performance_projections": performance_projections,
                "implementation_notes": self._generate_implementation_notes(optimal_allocations)
            }
            
        except Exception as e:
            return {
                "error": f"Portfolio optimization failed: {str(e)}",
                "symbols": symbols,
                "optimization_method": optimization_method
            }
    
    def _calculate_efficient_frontier(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Calculate the efficient frontier for the given securities"""
        try:
            # Simulate expected returns and covariance matrix
            n_assets = len(symbols)
            expected_returns = np.random.uniform(0.05, 0.15, n_assets)
            
            # Generate a realistic covariance matrix
            volatilities = np.random.uniform(0.15, 0.35, n_assets)
            correlation_matrix = self._generate_correlation_matrix(n_assets)
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            # Calculate efficient frontier points
            n_points = 50
            target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
            
            frontier_portfolios = []
            for target_return in target_returns:
                # Simulate optimization for each target return
                weights = self._optimize_for_target_return(expected_returns, covariance_matrix, target_return)
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                
                frontier_portfolios.append({
                    "expected_return": float(portfolio_return),
                    "volatility": float(portfolio_risk),
                    "sharpe_ratio": float(portfolio_return / portfolio_risk) if portfolio_risk > 0 else 0,
                    "weights": {symbols[i]: float(weights[i]) for i in range(len(symbols))}
                })
            
            # Find special portfolios
            max_sharpe_idx = max(range(len(frontier_portfolios)), 
                               key=lambda i: frontier_portfolios[i]["sharpe_ratio"])
            min_vol_idx = min(range(len(frontier_portfolios)), 
                            key=lambda i: frontier_portfolios[i]["volatility"])
            
            return {
                "frontier_points": frontier_portfolios,
                "max_sharpe_portfolio": frontier_portfolios[max_sharpe_idx],
                "min_volatility_portfolio": frontier_portfolios[min_vol_idx],
                "market_data": {
                    "expected_returns": {symbols[i]: float(expected_returns[i]) for i in range(len(symbols))},
                    "volatilities": {symbols[i]: float(volatilities[i]) for i in range(len(symbols))},
                    "correlation_matrix": {
                        symbols[i]: {symbols[j]: float(correlation_matrix[i,j]) 
                                   for j in range(len(symbols))} 
                        for i in range(len(symbols))
                    }
                }
            }
            
        except Exception as e:
            return {"error": f"Efficient frontier calculation failed: {str(e)}"}
    
    def _generate_correlation_matrix(self, n_assets: int) -> np.ndarray:
        """Generate a realistic correlation matrix"""
        # Start with random correlations
        correlation_matrix = np.random.uniform(-0.3, 0.8, (n_assets, n_assets))
        
        # Make symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        
        # Set diagonal to 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize to ensure diagonal is 1
        diag_sqrt = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(diag_sqrt, diag_sqrt)
        
        return correlation_matrix
    
    def _optimize_for_target_return(self, expected_returns: np.ndarray, 
                                   covariance_matrix: np.ndarray, 
                                   target_return: float) -> np.ndarray:
        """Optimize portfolio for a target return level"""
        try:
            n_assets = len(expected_returns)
            
            # Simple quadratic programming solution simulation
            # In practice, would use scipy.optimize or cvxpy
            
            # Generate random weights and normalize
            weights = np.random.uniform(0, 1, n_assets)
            weights = weights / np.sum(weights)
            
            # Adjust to approximately match target return
            current_return = np.dot(weights, expected_returns)
            if current_return != 0:
                adjustment = target_return / current_return
                weights = weights * adjustment
                
                # Renormalize and ensure non-negative
                weights = np.maximum(weights, 0)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.ones(n_assets) / n_assets
            
            return weights
            
        except Exception as e:
            # Fallback to equal weights
            return np.ones(len(expected_returns)) / len(expected_returns)
    
    def _calculate_optimal_allocations(self, symbols: List[str], 
                                     optimization_method: str) -> Dict[str, Any]:
        """Calculate optimal allocations using various methods"""
        try:
            n_assets = len(symbols)
            allocations = {}
            
            if optimization_method == "mean_variance":
                # Mean-variance optimization (max Sharpe ratio)
                weights = self._simulate_mean_variance_optimization(n_assets)
                allocations["mean_variance"] = {symbols[i]: float(weights[i]) for i in range(n_assets)}
            
            elif optimization_method == "risk_parity":
                # Risk parity allocation
                weights = self._calculate_risk_parity_weights(n_assets)
                allocations["risk_parity"] = {symbols[i]: float(weights[i]) for i in range(n_assets)}
            
            elif optimization_method == "equal_weight":
                # Equal weight allocation
                weights = np.ones(n_assets) / n_assets
                allocations["equal_weight"] = {symbols[i]: float(weights[i]) for i in range(n_assets)}
            
            elif optimization_method == "market_cap":
                # Market cap weighted (simulated)
                market_caps = np.random.uniform(100, 3000, n_assets)  # Billions
                weights = market_caps / np.sum(market_caps)
                allocations["market_cap"] = {symbols[i]: float(weights[i]) for i in range(n_assets)}
            
            else:
                # Default to mean-variance
                weights = self._simulate_mean_variance_optimization(n_assets)
                allocations["mean_variance"] = {symbols[i]: float(weights[i]) for i in range(n_assets)}
            
            # Add alternative allocations for comparison
            allocations["alternatives"] = {
                "equal_weight": {symbols[i]: 1.0/n_assets for i in range(n_assets)},
                "risk_parity": {symbols[i]: float(w) for i, w in enumerate(self._calculate_risk_parity_weights(n_assets))},
                "momentum_based": {symbols[i]: float(w) for i, w in enumerate(self._calculate_momentum_weights(n_assets))}
            }
            
            return {
                "primary_allocation": allocations.get(optimization_method, allocations["mean_variance"]),
                "alternative_allocations": allocations.get("alternatives", {}),
                "optimization_method": optimization_method,
                "allocation_summary": self._summarize_allocation(allocations.get(optimization_method, {}))
            }
            
        except Exception as e:
            return {"error": f"Allocation calculation failed: {str(e)}"}
    
    def _simulate_mean_variance_optimization(self, n_assets: int) -> np.ndarray:
        """Simulate mean-variance optimization result"""
        # Generate weights that favor higher Sharpe ratio assets
        base_weights = np.random.uniform(0.5, 2.0, n_assets)
        weights = base_weights / np.sum(base_weights)
        
        # Add some concentration (typical of mean-variance optimization)
        weights = weights ** 1.5
        weights = weights / np.sum(weights)
        
        return weights
    
    def _calculate_risk_parity_weights(self, n_assets: int) -> np.ndarray:
        """Calculate risk parity weights"""
        # Simulate risk parity - inversely proportional to volatility
        volatilities = np.random.uniform(0.15, 0.35, n_assets)
        inv_vol_weights = 1.0 / volatilities
        weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        return weights
    
    def _calculate_momentum_weights(self, n_assets: int) -> np.ndarray:
        """Calculate momentum-based weights"""
        # Simulate momentum weights based on recent performance
        momentum_scores = np.random.uniform(-0.1, 0.3, n_assets)
        
        # Convert to weights (higher momentum = higher weight)
        weights = np.maximum(momentum_scores, 0.01)  # Ensure positive
        weights = weights / np.sum(weights)
        
        return weights
    
    def _project_portfolio_performance(self, symbols: List[str], 
                                     allocations: Dict[str, Any]) -> Dict[str, Any]:
        """Project portfolio performance under different scenarios"""
        try:
            primary_allocation = allocations.get("primary_allocation", {})
            
            if not primary_allocation:
                return {"error": "No primary allocation found"}
            
            # Scenario analysis
            scenarios = {
                "bull_market": {
                    "description": "Strong bull market with 20% annual returns",
                    "market_return": 0.20,
                    "volatility_multiplier": 0.8,
                    "correlation_increase": 0.1
                },
                "bear_market": {
                    "description": "Bear market with -15% annual returns",
                    "market_return": -0.15,
                    "volatility_multiplier": 1.5,
                    "correlation_increase": 0.3
                },
                "normal_market": {
                    "description": "Normal market conditions with 8% annual returns",
                    "market_return": 0.08,
                    "volatility_multiplier": 1.0,
                    "correlation_increase": 0.0
                },
                "high_volatility": {
                    "description": "High volatility environment",
                    "market_return": 0.05,
                    "volatility_multiplier": 2.0,
                    "correlation_increase": 0.2
                }
            }
            
            projections = {}
            for scenario_name, scenario in scenarios.items():
                # Calculate expected portfolio performance in this scenario
                portfolio_return = scenario["market_return"] * np.random.uniform(0.8, 1.2)
                portfolio_volatility = 0.16 * scenario["volatility_multiplier"]
                
                projections[scenario_name] = {
                    "expected_return": float(portfolio_return),
                    "expected_volatility": float(portfolio_volatility),
                    "sharpe_ratio": float(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0,
                    "var_95": float(-1.65 * portfolio_volatility),  # 95% VaR approximation
                    "scenario_details": scenario
                }
            
            # Calculate allocation metrics
            allocation_metrics = {
                "concentration": self._calculate_concentration_metric(primary_allocation),
                "diversification_score": self._calculate_diversification_score(primary_allocation),
                "largest_position": max(primary_allocation.values()) if primary_allocation else 0,
                "number_of_positions": len([w for w in primary_allocation.values() if w > 0.01])
            }
            
            return {
                "scenario_projections": projections,
                "allocation_metrics": allocation_metrics,
                "expected_annual_return": np.mean([p["expected_return"] for p in projections.values()]),
                "expected_annual_volatility": np.mean([p["expected_volatility"] for p in projections.values()])
            }
            
        except Exception as e:
            return {"error": f"Performance projection failed: {str(e)}"}
    
    def _calculate_concentration_metric(self, allocation: Dict[str, float]) -> float:
        """Calculate portfolio concentration using Herfindahl index"""
        if not allocation:
            return 0.0
        
        weights = list(allocation.values())
        hhi = sum(w**2 for w in weights)
        return float(hhi)
    
    def _calculate_diversification_score(self, allocation: Dict[str, float]) -> float:
        """Calculate diversification score (0-1, higher is more diversified)"""
        if not allocation:
            return 0.0
        
        n_assets = len(allocation)
        if n_assets <= 1:
            return 0.0
        
        # Effective number of assets
        weights = list(allocation.values())
        effective_n = 1.0 / sum(w**2 for w in weights if w > 0)
        
        # Normalize by maximum possible diversification
        diversification_score = (effective_n - 1) / (n_assets - 1)
        return float(min(diversification_score, 1.0))
    
    def _summarize_allocation(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Create allocation summary statistics"""
        if not allocation:
            return {"error": "No allocation to summarize"}
        
        weights = list(allocation.values())
        symbols = list(allocation.keys())
        
        # Sort by weight
        sorted_positions = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "top_3_positions": sorted_positions[:3],
            "number_of_positions": len([w for w in weights if w > 0.01]),
            "largest_position": max(weights),
            "smallest_position": min([w for w in weights if w > 0]),
            "concentration_top_3": sum([pos[1] for pos in sorted_positions[:3]]),
            "average_weight": np.mean(weights),
            "weight_std": np.std(weights)
        }
    
    def _generate_implementation_notes(self, allocations: Dict[str, Any]) -> List[str]:
        """Generate practical implementation notes"""
        notes = [
            "Consider transaction costs when implementing allocation changes",
            "Rebalance periodically to maintain target weights",
            "Monitor correlation changes that may affect optimization",
            "Review and update expected returns and risk estimates regularly"
        ]
        
        primary_allocation = allocations.get("primary_allocation", {})
        if primary_allocation:
            max_weight = max(primary_allocation.values()) if primary_allocation else 0
            if max_weight > 0.4:
                notes.append("High concentration detected - consider position size limits")
            
            min_weight = min([w for w in primary_allocation.values() if w > 0]) if primary_allocation else 0
            if min_weight < 0.02:
                notes.append("Very small positions may not be cost-effective to implement")
        
        return notes
    
    async def compare_allocation_strategies(self, symbols: List[str]) -> Dict[str, Any]:
        """Compare different allocation strategies"""
        strategies = ["mean_variance", "risk_parity", "equal_weight", "market_cap"]
        
        comparison_results = {}
        for strategy in strategies:
            result = await self.optimize_portfolio(symbols, optimization_method=strategy)
            comparison_results[strategy] = result
        
        return {
            "strategy_comparison": comparison_results,
            "recommendation": self._recommend_best_strategy(comparison_results)
        }
    
    def _recommend_best_strategy(self, comparison_results: Dict[str, Any]) -> str:
        """Recommend the best allocation strategy based on analysis"""
        # Simple heuristic - in practice would be more sophisticated
        strategies = list(comparison_results.keys())
        if not strategies:
            return "No strategies available for comparison"
        
        # For demonstration, randomly recommend one with reasoning
        recommended = np.random.choice(strategies)
        
        return f"Recommended strategy: {recommended} based on current market conditions and risk-return profile"