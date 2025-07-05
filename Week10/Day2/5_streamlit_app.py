import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
FASTAPI_URL = "http://localhost:9232"

# Page configuration
st.set_page_config(
    page_title="MCP Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
    .error-box {
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #f44336;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4caf50;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the FastAPI service is running"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def get_statistics():
    """Get calculation statistics"""
    try:
        response = requests.get(f"{FASTAPI_URL}/statistics")
        if response.status_code == 200:
            return response.json()["statistics"]
        return None
    except:
        return None

def get_summary():
    """Get calculation summary"""
    try:
        response = requests.get(f"{FASTAPI_URL}/summary")
        if response.status_code == 200:
            return response.json()["summary"]
        return None
    except:
        return None

def get_last_result():
    """Get last calculation result"""
    try:
        response = requests.get(f"{FASTAPI_URL}/last_result")
        if response.status_code == 200:
            return response.json()["last_result"]
        return None
    except:
        return None

def clear_history():
    """Clear calculation history"""
    try:
        response = requests.post(f"{FASTAPI_URL}/clear_history")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def calculate_expression(expression):
    """Send calculation request to FastAPI"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/calculate",
            json={"expression": expression},
            timeout=10
        )
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return False, str(e)

def batch_calculate(expressions):
    """Send batch calculation request"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/calculate/batch",
            json=expressions,
            timeout=30
        )
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return False, str(e)

# Initialize session state
if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []

# Header
st.title("🧮 MCP Calculator")
st.markdown("*Natural Language Calculator with Memory and Context*")

# Check API health
health_status, health_data = check_api_health()

if not health_status:
    st.error("❌ FastAPI service is not running. Please start the service on http://localhost:8000")
    st.markdown("""
    <div class="error-box">
        <strong>To start the FastAPI service:</strong><br>
        1. Make sure your MCP server is running on localhost:9321<br>
        2. Run: <code>uvicorn main:app --reload --host 0.0.0.0 --port 8000</code><br>
        3. Or run: <code>python run_server.py</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Sidebar - Status and Controls
with st.sidebar:
    st.header("📊 Service Status")

    if health_data:
        st.success("✅ FastAPI Service: Connected")
        st.info(f"🤖 Mistral AI: {health_data.get('mistral_ai', 'Unknown')}")
        st.info(f"📡 MCP Server: {health_data.get('mcp_server', 'Unknown')}")

        if 'session_stats' in health_data:
            stats = health_data['session_stats']
            st.metric("Total Calculations", stats.get('total_calculations', 0))

            last_result = stats.get('last_result')
            if last_result and last_result != "None":
                st.metric("Last Result", last_result)

    st.divider()

    # Quick Actions
    st.header("⚡ Quick Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            success, result = clear_history()
            if success:
                st.success("History cleared!")
                st.session_state.calculation_history = []
                st.rerun()
            else:
                st.error("Failed to clear history")

    with col2:
        if st.button("🔄 Refresh", type="secondary", use_container_width=True):
            st.rerun()

    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh stats", value=False)

    # Example expressions
    st.header("💡 Example Expressions")
    examples = [
        "What is 25 times 4?",
        "Calculate 100 divided by 5",
        "Add 15 and 35 together",
        "Subtract 12 from 50",
        "Add 10 to the result",
        "Multiply the answer by 3",
        "Divide the previous result by 2"
    ]

    for example in examples:
        if st.button(f"📝 {example}", key=f"example_{hash(example)}", use_container_width=True):
            st.session_state.single_calc = example
            st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Natural Language Calculator")

    # Single calculation
    with st.container():
        st.subheader("Single Calculation")

        # Get current context
        last_result = get_last_result()
        if last_result and last_result.strip() and last_result != "None":
            st.markdown(f"""
            <div class="info-box">
                📝 <strong>Previous result:</strong> {last_result}
            </div>
            """, unsafe_allow_html=True)

        expression = st.text_input(
            "Enter your calculation:",
            placeholder="e.g., 'What is 25 times 4?', 'Add 10 to the result', 'Divide the previous answer by 2'",
            key="single_calc"
        )

        col_calc, col_clear = st.columns([3, 1])

        with col_calc:
            if st.button("🚀 Calculate", type="primary", use_container_width=True):
                if expression:
                    with st.spinner("Calculating..."):
                        success, result = calculate_expression(expression)

                        if success:
                            calc_result = result["results"][0]
                            prev_result = result.get("previous_result", "None")

                            # Display result
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>✅ Result:</strong> {calc_result['result']}<br>
                                <strong>🔧 Operation:</strong> {calc_result['operation']}({calc_result['arguments']['a']}, {calc_result['arguments']['b']})<br>
                                <strong>📝 Previous:</strong> {prev_result}
                            </div>
                            """, unsafe_allow_html=True)

                            # Add to history
                            st.session_state.calculation_history.append({
                                "expression": expression,
                                "result": calc_result['result'],
                                "operation": calc_result['operation'],
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })

                        else:
                            st.markdown(f"""
                            <div class="error-box">
                                <strong>❌ Error:</strong> {result}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter a calculation expression")

        with col_clear:
            if st.button("🧹 Clear Input", use_container_width=True):
                st.session_state.single_calc = ""
                st.rerun()

    st.divider()

    # Batch calculations
    with st.container():
        st.subheader("🔄 Batch Calculations")
        st.write("Enter multiple calculations (one per line) to process sequentially:")

        batch_text = st.text_area(
            "Batch calculations:",
            placeholder="add 10 and 5\nsubtract 3 from the result\nmultiply the answer by 2",
            height=120,
            key="batch_calc"
        )

        col_batch, col_preset, col_clear_batch = st.columns([2, 1, 1])

        with col_batch:
            if st.button("⚡ Run Batch", type="primary", use_container_width=True):
                if batch_text:
                    expressions = [line.strip() for line in batch_text.split('\n') if line.strip()]

                    with st.spinner(f"Processing {len(expressions)} calculations..."):
                        success, result = batch_calculate(expressions)

                        if success:
                            batch_results = result["batch_results"]

                            st.write("**📊 Batch Results:**")
                            for i, calc in enumerate(batch_results, 1):
                                if calc["success"]:
                                    st.markdown(f"""
                                    <div class="success-box">
                                        <strong>{i}.</strong> '{calc['expression']}'<br>
                                        <strong>Result:</strong> {calc['result']}<br>
                                        <strong>Operation:</strong> {calc['operation']}({calc['arguments']['a']}, {calc['arguments']['b']})<br>
                                        <strong>Previous:</strong> {calc['previous_result']}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="error-box">
                                        <strong>{i}.</strong> '{calc['expression']}' - Error: {calc['error']}
                                    </div>
                                    """, unsafe_allow_html=True)

                                # Add successful calculations to history
                                if calc["success"]:
                                    st.session_state.calculation_history.append({
                                        "expression": calc['expression'],
                                        "result": calc['result'],
                                        "operation": calc['operation'],
                                        "timestamp": datetime.now().strftime("%H:%M:%S")
                                    })
                        else:
                            st.error(f"Batch calculation failed: {result}")
                else:
                    st.warning("Please enter batch calculations")

        with col_preset:
            if st.button("📋 Load Preset", use_container_width=True):
                preset_batch = """add 20 and 15
subtract 10 from the result
multiply the answer by 3
divide the result by 2
add 5 to the final answer"""
                st.session_state.batch_calc = preset_batch
                st.rerun()

        with col_clear_batch:
            if st.button("🧹 Clear Batch", use_container_width=True):
                st.session_state.batch_calc = ""
                st.rerun()

with col2:
    st.header("📈 Session Analytics")

    # Real-time statistics
    stats = get_statistics()
    if stats:
        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            st.metric("Total Calculations", stats.get('total_calculations', 0))

        with col_stat2:
            st.metric("Last Operation", stats.get('last_operation', 'None'))

        if stats.get('session_start'):
            try:
                session_start = datetime.fromisoformat(stats['session_start'].replace('Z', '+00:00'))
                duration = datetime.now() - session_start.replace(tzinfo=None)
                st.metric("Session Duration", f"{duration.seconds//60}m {duration.seconds%60}s")
            except:
                st.metric("Session Duration", "Unknown")

    st.divider()

    # Summary and charts
    summary = get_summary()
    if summary and summary.get('total_calculations', 0) > 0:
        st.subheader("📊 Operation Breakdown")

        # Operations pie chart
        if summary.get('operations_breakdown'):
            ops_data = summary['operations_breakdown']
            fig_pie = px.pie(
                values=list(ops_data.values()),
                names=list(ops_data.keys()),
                title="Operations Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Results statistics
        if summary.get('results'):
            results_stats = summary['results']
            st.subheader("📊 Results Statistics")

            col_min, col_max = st.columns(2)
            with col_min:
                st.metric("Min", f"{results_stats['min']:.2f}")
            with col_max:
                st.metric("Max", f"{results_stats['max']:.2f}")

            st.metric("Average", f"{results_stats['average']:.2f}")

            # Last 5 results chart
            if results_stats.get('last_5_results') and len(results_stats['last_5_results']) > 1:
                st.subheader("📈 Recent Results")
                last_results = results_stats['last_5_results']

                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=list(range(1, len(last_results) + 1)),
                    y=last_results,
                    mode='lines+markers',
                    name='Results',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                fig_line.update_layout(
                    title="Last 5 Results Trend",
                    xaxis_title="Calculation #",
                    yaxis_title="Result",
                    showlegend=False,
                    height=250
                )
                st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("📝 No calculations yet. Start by entering a calculation above!")

    # Recent history from session
    if st.session_state.calculation_history:
        st.subheader("🕐 Recent Session History")

        # Display last 5 calculations
        recent_history = st.session_state.calculation_history[-5:]
        for i, calc in enumerate(reversed(recent_history)):
            with st.expander(f"{calc['timestamp']}: {calc['result']}", expanded=False):
                st.write(f"**Expression:** {calc['expression']}")
                st.write(f"**Operation:** {calc['operation']}")
                st.write(f"**Result:** {calc['result']}")

# Footer
st.divider()
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("🔗 **[FastAPI Docs](http://localhost:8000/docs)**")

with col_footer2:
    st.markdown("📡 **[Health Check](http://localhost:8000/health)**")

with col_footer3:
    st.write(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")

# Auto-refresh (if enabled)
if auto_refresh:
    # Add a small delay and rerun to refresh data
    import time
    time.sleep(1)
    st.rerun()