"""
Financial Utilities - Common Financial Calculations and Metrics
Supporting the Financial Modeling Team
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FinancialCalculator:
    """
    Comprehensive financial calculations and metrics utility class
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # Default 2% risk-free rate
        self.trading_days_per_year = 252
    
    def calculate_returns(self, prices: np.ndarray, method: str = "simple") -> np.ndarray:
        """
        Calculate returns from price series
        
        Args:
            prices: Array of prices
            method: 'simple' or 'log' returns
            
        Returns:
            Array of returns
        """
        if len(prices) < 2:
            return np.array([])
        
        if method == "log":
            return np.diff(np.log(prices))
        else:  # simple returns
            return np.diff(prices) / prices[:-1]
    
    def calculate_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns)
        
        Args:
            returns: Array of returns
            annualize: Whether to annualize the volatility
            
        Returns:
            Volatility value
        """
        if len(returns) < 2:
            return 0.0
        
        vol = np.std(returns, ddof=1)
        
        if annualize:
            vol *= np.sqrt(self.trading_days_per_year)
        
        return float(vol)
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (if None, uses default)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        excess_returns = returns - (rf_rate / self.trading_days_per_year)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year))
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return)
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (if None, uses default)
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        excess_returns = returns - (rf_rate / self.trading_days_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        
        return float(np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year))
    
    def calculate_max_drawdown(self, prices: np.ndarray) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics
        
        Args:
            prices: Array of prices
            
        Returns:
            Dictionary with drawdown metrics
        """
        if len(prices) < 2:
            return {"max_drawdown": 0.0, "drawdown_duration": 0, "recovery_time": 0}
        
        # Calculate running maximum
        peak = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - peak) / peak
        
        max_drawdown = float(np.min(drawdown))
        max_dd_idx = np.argmin(drawdown)
        
        # Find the peak before max drawdown
        peak_idx = np.argmax(peak[:max_dd_idx + 1])
        
        # Calculate duration (from peak to trough)
        drawdown_duration = max_dd_idx - peak_idx
        
        # Calculate recovery time (from trough to recovery)
        recovery_idx = np.where(prices[max_dd_idx:] >= peak[max_dd_idx])[0]
        recovery_time = recovery_idx[0] if len(recovery_idx) > 0 else len(prices) - max_dd_idx
        
        return {
            "max_drawdown": max_drawdown,
            "drawdown_duration": int(drawdown_duration),
            "recovery_time": int(recovery_time),
            "peak_date_idx": int(peak_idx),
            "trough_date_idx": int(max_dd_idx)
        }
    
    def calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        Calculate beta (systematic risk measure)
        
        Args:
            asset_returns: Returns of the asset
            market_returns: Returns of the market benchmark
            
        Returns:
            Beta value
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 1.0  # Default beta
        
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns, ddof=1)
        
        if market_variance == 0:
            return 1.0
        
        return float(covariance / market_variance)
    
    def calculate_alpha(self, asset_returns: np.ndarray, market_returns: np.ndarray, 
                       risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate alpha (excess return over CAPM prediction)
        
        Args:
            asset_returns: Returns of the asset
            market_returns: Returns of the market benchmark  
            risk_free_rate: Risk-free rate
            
        Returns:
            Alpha value (annualized)
        """
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 2:
            return 0.0
        
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        daily_rf = rf_rate / self.trading_days_per_year
        
        beta = self.calculate_beta(asset_returns, market_returns)
        
        asset_excess = np.mean(asset_returns) - daily_rf
        market_excess = np.mean(market_returns) - daily_rf
        
        alpha = asset_excess - beta * market_excess
        
        return float(alpha * self.trading_days_per_year)  # Annualize
    
    def calculate_information_ratio(self, asset_returns: np.ndarray, 
                                  benchmark_returns: np.ndarray) -> float:
        """
        Calculate information ratio (active return / tracking error)
        
        Args:
            asset_returns: Returns of the asset/portfolio
            benchmark_returns: Returns of the benchmark
            
        Returns:
            Information ratio
        """
        if len(asset_returns) != len(benchmark_returns) or len(asset_returns) < 2:
            return 0.0
        
        active_returns = asset_returns - benchmark_returns
        active_return = np.mean(active_returns)
        tracking_error = np.std(active_returns, ddof=1)
        
        if tracking_error == 0:
            return 0.0
        
        return float(active_return / tracking_error * np.sqrt(self.trading_days_per_year))
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95, 
                     method: str = "historical") -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'modified'
            
        Returns:
            VaR value (negative number representing loss)
        """
        if len(returns) < 2:
            return 0.0
        
        if method == "historical":
            return float(np.percentile(returns, (1 - confidence_level) * 100))
        
        elif method == "parametric":
            from scipy.stats import norm
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            z_score = norm.ppf(1 - confidence_level)
            return float(mean_return + z_score * std_return)
        
        elif method == "modified":
            # Cornish-Fisher modification for skewness and kurtosis
            from scipy.stats import skew, kurtosis, norm
            
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            skewness = skew(returns)
            kurt = kurtosis(returns)
            
            z = norm.ppf(1 - confidence_level)
            
            # Cornish-Fisher expansion
            z_cf = (z + (z**2 - 1) * skewness / 6 + 
                   (z**3 - 3*z) * kurt / 24 - 
                   (2*z**3 - 5*z) * skewness**2 / 36)
            
            return float(mean_return + z_cf * std_return)
        
        else:
            raise ValueError("Method must be 'historical', 'parametric', or 'modified'")
    
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value
        """
        if len(returns) < 2:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level, method="historical")
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return float(np.mean(tail_returns))
    
    def calculate_calmar_ratio(self, returns: np.ndarray, prices: np.ndarray) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown)
        
        Args:
            returns: Array of returns
            prices: Array of prices for drawdown calculation
            
        Returns:
            Calmar ratio
        """
        if len(returns) < 2 or len(prices) < 2:
            return 0.0
        
        annual_return = np.mean(returns) * self.trading_days_per_year
        max_dd = self.calculate_max_drawdown(prices)["max_drawdown"]
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return float(annual_return / abs(max_dd))
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio-level metrics
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            
        Returns:
            Dictionary of portfolio metrics
        """
        if len(weights) != len(expected_returns) or len(weights) != covariance_matrix.shape[0]:
            return {}
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        # Portfolio return
        portfolio_return = float(np.dot(weights, expected_returns))
        
        # Portfolio variance and volatility
        portfolio_variance = float(np.dot(weights, np.dot(covariance_matrix, weights)))
        portfolio_volatility = float(np.sqrt(portfolio_variance))
        
        # Sharpe ratio (assuming risk-free rate)
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = float(excess_return / portfolio_volatility) if portfolio_volatility > 0 else 0.0
        
        return {
            "expected_return": portfolio_return,
            "volatility": portfolio_volatility,
            "variance": portfolio_variance,
            "sharpe_ratio": sharpe_ratio,
            "weights": weights.tolist()
        }
    
    def calculate_efficient_frontier_point(self, expected_returns: np.ndarray,
                                         covariance_matrix: np.ndarray,
                                         target_return: float) -> Dict[str, Any]:
        """
        Calculate a single point on the efficient frontier
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            target_return: Target portfolio return
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            # Objective function: minimize portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(covariance_matrix, weights))
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # weights sum to 1
                {'type': 'eq', 'fun': lambda weights: np.dot(weights, expected_returns) - target_return}  # target return
            ]
            
            # Bounds (no short selling)
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                portfolio_metrics = self.calculate_portfolio_metrics(
                    optimal_weights, expected_returns, covariance_matrix
                )
                return {
                    "success": True,
                    "weights": optimal_weights.tolist(),
                    "metrics": portfolio_metrics
                }
            else:
                return {"success": False, "error": "Optimization failed"}
                
        except ImportError:
            # Fallback if scipy not available
            return {
                "success": False, 
                "error": "scipy.optimize not available",
                "fallback_weights": (np.ones(len(expected_returns)) / len(expected_returns)).tolist()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def calculate_correlation_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix from returns
        
        Args:
            returns_matrix: Matrix where each column is returns for an asset
            
        Returns:
            Correlation matrix
        """
        return np.corrcoef(returns_matrix.T)
    
    def calculate_tracking_error(self, portfolio_returns: np.ndarray, 
                               benchmark_returns: np.ndarray) -> float:
        """
        Calculate tracking error (standard deviation of active returns)
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Annualized tracking error
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns, ddof=1)
        
        return float(tracking_error * np.sqrt(self.trading_days_per_year))
    
    def calculate_diversification_ratio(self, weights: np.ndarray, 
                                      volatilities: np.ndarray,
                                      correlation_matrix: np.ndarray) -> float:
        """
        Calculate diversification ratio
        
        Args:
            weights: Portfolio weights
            volatilities: Individual asset volatilities
            correlation_matrix: Asset correlation matrix
            
        Returns:
            Diversification ratio
        """
        if len(weights) != len(volatilities) or len(weights) != correlation_matrix.shape[0]:
            return 1.0
        
        # Weighted average volatility
        weighted_vol = np.dot(weights, volatilities)
        
        # Portfolio volatility
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        
        if portfolio_vol == 0:
            return 1.0
        
        return float(weighted_vol / portfolio_vol)
    
    def monte_carlo_var(self, returns: np.ndarray, confidence_level: float = 0.95,
                       n_simulations: int = 10000, time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate VaR using Monte Carlo simulation
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR and additional statistics
        """
        if len(returns) < 2:
            return {"var": 0.0, "cvar": 0.0}
        
        # Estimate parameters from historical data
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.normal(mean_return, std_return, 
                                        (n_simulations, time_horizon))
        
        # Calculate cumulative returns for each simulation
        cumulative_returns = np.prod(1 + random_returns, axis=1) - 1
        
        # Calculate VaR and CVaR
        var = float(np.percentile(cumulative_returns, (1 - confidence_level) * 100))
        cvar = float(np.mean(cumulative_returns[cumulative_returns <= var]))
        
        return {
            "var": var,
            "cvar": cvar,
            "mean_simulated_return": float(np.mean(cumulative_returns)),
            "std_simulated_return": float(np.std(cumulative_returns)),
            "n_simulations": n_simulations,
            "time_horizon": time_horizon
        }
    
    def black_litterman_weights(self, market_caps: np.ndarray, 
                              expected_returns: np.ndarray,
                              covariance_matrix: np.ndarray,
                              risk_aversion: float = 3.0,
                              tau: float = 0.05) -> Dict[str, Any]:
        """
        Calculate Black-Litterman optimal weights (simplified implementation)
        
        Args:
            market_caps: Market capitalizations for calculating market weights
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            tau: Uncertainty parameter
            
        Returns:
            Dictionary with Black-Litterman weights and metrics
        """
        try:
            # Market weights (proportional to market cap)
            market_weights = market_caps / np.sum(market_caps)
            
            # Implied equilibrium returns
            implied_returns = risk_aversion * np.dot(covariance_matrix, market_weights)
            
            # Black-Litterman formula (without views for simplicity)
            # In practice, you would incorporate investor views here
            uncertainty_matrix = tau * covariance_matrix
            
            # Calculate new expected returns (same as implied without views)
            bl_returns = implied_returns
            
            # Calculate optimal weights
            inv_cov = np.linalg.inv(covariance_matrix)
            optimal_weights = np.dot(inv_cov, bl_returns) / risk_aversion
            
            # Normalize weights
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            return {
                "success": True,
                "optimal_weights": optimal_weights.tolist(),
                "market_weights": market_weights.tolist(),
                "implied_returns": implied_returns.tolist(),
                "bl_returns": bl_returns.tolist()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_weights": (market_caps / np.sum(market_caps)).tolist()
            }