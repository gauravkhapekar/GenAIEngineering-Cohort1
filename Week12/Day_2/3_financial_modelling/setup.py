"""
Setup Script for Financial Modeling Team
Helps users get started with the phidata financial modeling system
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directory_structure():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data",
        "data/outputs",
        "logs",
        "agents",
        "utils"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        "phidata>=2.4.0",
        "openai>=1.0.0",
        "numpy>=1.24.0", 
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "yfinance>=0.2.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "python-dateutil>=2.8.0",
        "requests>=2.31.0",
        "jsonschema>=4.17.0"
    ]
    
    for package in requirements:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            return False
    
    return True

def check_api_keys():
    """Check if required API keys are set"""
    print("🔑 Checking API keys...")
    
    required_keys = {
        "OPENAI_API_KEY": "Required for AI agents",
        "ALPHA_VANTAGE_API_KEY": "Optional: For enhanced market data",
        "FRED_API_KEY": "Optional: For economic data"
    }
    
    missing_keys = []
    
    for key, description in required_keys.items():
        if os.getenv(key):
            print(f"  ✅ {key}: Set")
        else:
            print(f"  ❌ {key}: Not set ({description})")
            if key == "OPENAI_API_KEY":
                missing_keys.append(key)
    
    return len(missing_keys) == 0

def create_env_file_template():
    """Create a template .env file"""
    print("📝 Creating .env template...")
    
    env_template = """# Financial Modeling Team Environment Variables
# Copy this file to .env and fill in your API keys

# Required: OpenAI API Key for AI agents
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Alpha Vantage API Key for enhanced market data
# Get free key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Optional: FRED API Key for economic data
# Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here

# Optional: Quandl API Key for additional data sources
QUANDL_API_KEY=your_quandl_key_here
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("  ✅ Created .env.template file")
    print("  📝 Copy this to .env and add your API keys")

def run_basic_test():
    """Run a basic test to verify installation"""
    print("🧪 Running basic test...")
    
    try:
        # Test imports
        import numpy as np
        import pandas as pd
        from datetime import datetime
        
        # Test basic functionality
        test_data = np.random.randn(100)
        test_df = pd.DataFrame({"values": test_data})
        
        print("  ✅ Basic imports working")
        print("  ✅ NumPy and Pandas functional")
        
        # Test if we can import our modules (if they exist)
        try:
            from utils.financial_utils import FinancialCalculator
            calc = FinancialCalculator()
            print("  ✅ Financial utilities accessible")
        except ImportError:
            print("  ⚠️  Financial utilities not found (expected if running setup first)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic test failed: {e}")
        return False

def create_quickstart_example():
    """Create a quickstart example file"""
    print("📚 Creating quickstart example...")
    
    quickstart_code = '''"""
Financial Modeling Team - Quick Start Example
Run this after completing setup to test the system
"""

import asyncio
from main import FinancialModelingTeam

async def quickstart_demo():
    """Quick demonstration of the financial modeling team"""
    
    print("🏦 Financial Modeling Team - Quick Start Demo")
    print("=" * 50)
    
    # Initialize the team
    try:
        team = FinancialModelingTeam()
        print("✅ Team initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize team: {e}")
        print("💡 Make sure your OPENAI_API_KEY is set in .env file")
        return
    
    # Test portfolio symbols
    test_portfolio = ["AAPL", "GOOGL", "MSFT"]
    
    print(f"\\n📊 Testing with portfolio: {test_portfolio}")
    
    # Get team status
    status = team.get_team_status()
    print("\\n👥 Team Status:")
    for agent, status_msg in status.items():
        print(f"  {agent}: {status_msg}")
    
    # Quick market insight
    try:
        print(f"\\n💡 Getting quick market insight...")
        insight = await team.quick_market_insight(test_portfolio)
        print("Market Insight:")
        print(insight[:500] + "..." if len(insight) > 500 else insight)
        
    except Exception as e:
        print(f"❌ Market insight failed: {e}")
    
    print("\\n✅ Quick start demo completed!")
    print("\\n🎯 Next steps:")
    print("  1. Run full analysis: await team.analyze_portfolio(symbols)")
    print("  2. Generate reports: await team.generate_comprehensive_report(symbols)")
    print("  3. Customize agents in the agents/ directory")
    print("  4. Add your own data sources in utils/data_sources.py")

if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("💡 Install python-dotenv for .env file support: pip install python-dotenv")
    
    # Run the demo
    asyncio.run(quickstart_demo())
'''
    
    with open("quickstart.py", "w") as f:
        f.write(quickstart_code)
    
    print("  ✅ Created quickstart.py")

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup completed!")
    print("\n📝 Next Steps:")
    print("1. Set up your API keys:")
    print("   - Copy .env.template to .env")
    print("   - Add your OpenAI API key (required)")
    print("   - Add other API keys (optional)")
    
    print("\n2. Test the installation:")
    print("   python quickstart.py")
    
    print("\n3. Run the full demo:")
    print("   python main.py")
    
    print("\n4. Explore the codebase:")
    print("   - agents/: Specialized AI agents")
    print("   - utils/: Financial calculations and utilities")  
    print("   - data/: Sample data and outputs")
    print("   - config.py: Configuration settings")
    
    print("\n📚 Documentation:")
    print("   - README.md: Complete documentation")
    print("   - requirements.txt: Package dependencies")
    print("   - config.py: Configuration options")
    
    print("\n🔧 Customization:")
    print("   - Modify agents in agents/ directory")
    print("   - Add data sources in utils/data_sources.py")
    print("   - Adjust settings in config.py")
    
    print("\n⚠️  Important Notes:")
    print("   - This is a demonstration system")
    print("   - Not for actual investment decisions")
    print("   - Always validate financial calculations")
    print("   - Consult financial professionals for real trading")

def main():
    """Main setup function"""
    print("🏦 Financial Modeling Team Setup")
    print("=" * 50)
    print("Setting up your phidata financial modeling environment...\n")
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Create directories
    create_directory_structure()
    
    # Step 3: Install packages
    if not install_requirements():
        print("❌ Package installation failed. Please check your internet connection and try again.")
        return False
    
    # Step 4: Check API keys
    api_keys_ok = check_api_keys()
    
    # Step 5: Create template files
    create_env_file_template()
    create_quickstart_example()
    
    # Step 6: Run basic test
    if not run_basic_test():
        print("❌ Basic tests failed. Please check the installation.")
        return False
    
    # Step 7: Print next steps
    print_next_steps()
    
    if not api_keys_ok:
        print("\n⚠️  Setup completed with warnings: API keys not configured")
        print("Please set your OpenAI API key before running the system.")
    else:
        print("\n✅ Setup completed successfully!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)