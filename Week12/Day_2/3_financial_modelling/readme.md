# Financial Modeling Team - AI Agent System

A sophisticated team of AI agents built with phidata for comprehensive financial modeling, analysis, and portfolio management.

## 🎯 Overview

This system consists of specialized AI agents that work together to provide comprehensive financial modeling capabilities:

- **Data Analyst Agent**: Gathers and processes financial data from various sources
- **Risk Assessment Agent**: Evaluates portfolio and investment risks using various models
- **Portfolio Optimizer Agent**: Optimizes asset allocation and portfolio composition
- **Financial Forecaster Agent**: Provides predictive analytics and forecasting
- **Report Generator Agent**: Creates comprehensive financial reports and visualizations
- **Team Coordinator**: Orchestrates the entire workflow and manages agent interactions

## 🚀 Features

- **Real-time Financial Data Processing**: Fetch and analyze market data
- **Risk Analysis**: VaR calculations, beta analysis, correlation studies
- **Portfolio Optimization**: Modern Portfolio Theory implementation
- **Predictive Modeling**: Time series forecasting and trend analysis
- **Automated Reporting**: Generate professional financial reports
- **Interactive Dashboard**: Web-based interface for team interaction

## 📁 Project Structure

```
financial_modeling_team/
├── README.md
├── requirements.txt
├── main.py                    # Entry point and team coordinator
├── config.py                  # Configuration settings
├── setup.py                   # Setup and installation script
├── agents/
│   ├── __init__.py
│   ├── data_analyst.py        # Data collection and preprocessing
│   ├── risk_assessor.py       # Risk analysis and metrics
│   ├── portfolio_optimizer.py # Portfolio optimization algorithms
│   ├── forecaster.py          # Predictive modeling
│   └── report_generator.py    # Report and visualization creation
├── utils/
│   ├── __init__.py
│   ├── financial_utils.py     # Financial calculation utilities
│   └── data_sources.py        # Data source configurations
└── data/
    ├── sample_data.json       # Sample financial data
    └── outputs/               # Generated reports and outputs
```

## 🛠️ Installation

1. Clone or download the project files
2. Run the setup script:

```bash
python setup.py
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.template` to `.env`
   - Add your OpenAI API key (required)
   - Add other API keys (optional)

5. Test the installation:

```bash
python quickstart.py
```

## 💡 Usage Examples

### Basic Portfolio Analysis
```python
import asyncio
from main import FinancialModelingTeam

async def main():
    # Initialize the team
    team = FinancialModelingTeam()
    
    # Analyze a portfolio
    portfolio = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    analysis = await team.analyze_portfolio(portfolio)
    
    print(analysis)

asyncio.run(main())
```

### Risk Assessment
```python
# Perform comprehensive risk analysis
risk_report = await team.assess_portfolio_risk(
    portfolio=["AAPL", "GOOGL", "MSFT"], 
    timeframe="1Y"
)
```

### Generate Reports
```python
# Create a full financial report
report = await team.generate_comprehensive_report(
    portfolio=["AAPL", "GOOGL", "MSFT"],
    output_format="html"
)
```

## 🧠 Agent Capabilities

### Data Analyst Agent
- Fetches real-time and historical market data
- Cleans and preprocesses financial datasets
- Calculates basic financial metrics (returns, volatility, etc.)
- Performs correlation and statistical analysis

### Risk Assessment Agent
- Calculates Value at Risk (VaR) using multiple methods
- Performs beta analysis and market risk assessment
- Evaluates portfolio concentration and diversification
- Stress testing and scenario analysis

### Portfolio Optimizer Agent
- Implements Modern Portfolio Theory optimization
- Efficient frontier calculation
- Risk parity and equal weight strategies
- Constraint-based optimization

### Financial Forecaster Agent
- Time series forecasting using ARIMA and other models
- Trend analysis and technical indicators
- Monte Carlo simulations for price predictions
- Economic indicator analysis

### Report Generator Agent
- Creates professional PDF and HTML reports
- Interactive charts and visualizations
- Performance attribution analysis
- Executive summary generation

## 📊 Key Metrics Calculated

- **Returns**: Daily, monthly, annual returns
- **Risk Metrics**: Standard deviation, VaR, CVaR, beta
- **Ratios**: Sharpe ratio, Sortino ratio, information ratio
- **Portfolio Metrics**: Diversification ratio, maximum drawdown
- **Performance**: Alpha, tracking error, correlation matrix

## 🎨 Customization

The system is highly modular and can be customized:

1. **Add New Agents**: Create new agent classes inheriting from base agent
2. **Custom Metrics**: Add new financial calculations in `financial_utils.py`
3. **Data Sources**: Configure new data providers in `data_sources.py`
4. **Reporting**: Customize report templates and visualizations

## 🔧 Configuration

Key configuration options in `config.py`:
- Data source preferences
- Risk calculation methods
- Optimization constraints
- Report formatting options

## 📈 Sample Output

The system generates:
- Portfolio performance dashboards
- Risk analysis reports
- Optimization recommendations
- Forecast charts and predictions
- Executive summaries

## 🔑 API Keys Required

### Required
- **OpenAI API Key**: For AI agent functionality
  - Get it at: https://platform.openai.com/api-keys

### Optional
- **Alpha Vantage API Key**: For enhanced market data
  - Get free key at: https://www.alphavantage.co/support/#api-key
- **FRED API Key**: For economic data
  - Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run quick demo:
```bash
python quickstart.py
```

Run full demo:
```bash
python main.py
```

## 🤝 Contributing

This is a modular system designed for easy extension. Feel free to:
- Add new financial models
- Enhance existing agents
- Create new visualization types
- Improve data source integrations

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Errors**: Ensure your OpenAI API key is set
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. **Data Source Issues**: Check internet connection and API limits

### Getting Help

1. Check the configuration: `python config.py`
2. Run diagnostics: `python setup.py`
3. Review logs in `logs/` directory

## 📄 License

This project is provided as-is for educational and research purposes.

## ⚠️ Important Disclaimers

- This is a demonstration system for educational purposes
- Not intended for actual investment decisions
- Always validate financial calculations independently
- Consult with financial professionals before making investment decisions
- Past performance does not guarantee future results

## 🎯 Next Steps

1. **Customize Agents**: Modify agent behavior in the `agents/` directory
2. **Add Data Sources**: Extend data sources in `utils/data_sources.py`
3. **Create Custom Reports**: Build new report templates
4. **Implement New Models**: Add advanced financial models
5. **Scale the System**: Deploy on cloud infrastructure

---

**Built with phidata framework for intelligent agent orchestration**

For more information, visit: https://docs.phidata.com