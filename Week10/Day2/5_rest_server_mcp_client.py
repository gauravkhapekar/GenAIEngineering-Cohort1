from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client
from mistralai import Mistral
from contextlib import asynccontextmanager
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
# Pydantic models
class NaturalLanguageRequest(BaseModel):
    expression: str = Field(..., description="Natural language calculation request")

# Configuration
MCP_SERVER_URL = "http://localhost:9321/sse"

# Set your Mistral API key
if 'MISTRAL_KEY' not in os.environ:
    raise ValueError("MISTRAL_KEY environment variable is not set.")
mistral = Mistral(api_key=os.getenv('MISTRAL_KEY'))

# Global variables
available_tools = []
mistral_client = None

async def get_mcp_session():
    """Create and return an MCP client session"""
    return sse_client(MCP_SERVER_URL)

async def initialize_tools():
    """Initialize available tools from MCP server"""
    global available_tools
    try:
        async with await get_mcp_session() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                available_tools = tools_result.tools
                print(f"📊 Connected to MCP server! Found {len(available_tools)} tools")
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        raise

def initialize_mistral():
    """Initialize Mistral AI client"""
    global mistral_client
    if 'MISTRAL_KEY' not in os.environ:
        print("MISTRAL_KEY environment variable is not set.")

    mistral_client = Mistral(api_key=os.getenv('MISTRAL_KEY'))

    # if MISTRAL_KEY and MISTRAL_API_KEY != "your-mistral-api-key-here":
    #     mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    #     print("🤖 Mistral AI client initialized")
    # else:
    #     print("⚠️  Mistral AI not configured")

def create_mistral_tools():
    """Convert MCP tools to Mistral tool format"""
    mistral_tools = []
    for tool in available_tools:
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        if tool.name in ["add", "subtract", "multiply", "divide"]:
            parameters["properties"] = {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            }
            parameters["required"] = ["a", "b"]

        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters
            }
        }
        mistral_tools.append(tool_def)

    return mistral_tools

async def execute_mcp_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool call on the MCP server"""
    try:
        async with await get_mcp_session() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP tool execution failed: {str(e)}")

async def get_previous_result():
    """Get the last calculation result from MCP server"""
    try:
        async with await get_mcp_session() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_last_result", {})
                return result.content[0].text
    except Exception as e:
        return None

def generate_new_query(user_input: str, previous_result=None):
    """Generate a new query using Mistral with context of previous result"""
    if not mistral_client:
        return None

    mistral_tools = create_mistral_tools()

    context_message = f"previous result is {previous_result} and user_input is {user_input}. return operation and two relevant numbers to perform the calculation."

    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{
            "role": "system",
            "content": "You are a calculator assistant. Use the add, subtract, multiply, or divide functions to help the user with calculations. If the user refers to 'the result', 'previous answer', or similar, use the previous result provided."
        }, {
            "role": "user",
            "content": context_message
        }],
        tools=mistral_tools,
        tool_choice="auto"
    )
    return response.choices[0].message

async def read_mcp_resource(resource_uri: str) -> dict:
    """Read a resource from the MCP server"""
    try:
        async with await get_mcp_session() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resource = await session.read_resource(resource_uri)
                return json.loads(resource.contents[0].text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP resource read failed: {str(e)}")

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MCP Client FastAPI Service Starting...")
    await initialize_tools()
    initialize_mistral()
    print("✅ Service ready!")
    yield
    print("🛑 Service shutting down...")

# FastAPI app
app = FastAPI(
    title="MCP Calculator Client API",
    description="FastAPI service for natural language calculations, statistics, and summaries",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/calculate")
async def calculate_natural_language(request: NaturalLanguageRequest):
    """Process natural language calculation requests using Mistral AI with previous result context"""
    if not mistral_client:
        raise HTTPException(
            status_code=503,
            detail="Mistral AI client not configured. Please set MISTRAL_API_KEY."
        )

    try:
        # Get previous result for context
        previous_result = await get_previous_result()

        # Generate new query with context using the notebook's approach
        message = generate_new_query(request.expression, previous_result)

        if not message or not message.tool_calls:
            # Fallback to direct interpretation if generate_new_query fails
            mistral_tools = create_mistral_tools()

            response = mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a calculator assistant. Parse the user's expression and use the appropriate function to perform the calculation."
                    },
                    {
                        "role": "user",
                        "content": request.expression
                    }
                ],
                tools=mistral_tools,
                tool_choice="auto"
            )
            message = response.choices[0].message

        if not message.tool_calls:
            raise HTTPException(
                status_code=400,
                detail="Could not interpret the calculation request"
            )

        results = []
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name

            if isinstance(tool_call.function.arguments, str):
                args = json.loads(tool_call.function.arguments)
            else:
                args = tool_call.function.arguments

            result = await execute_mcp_tool(func_name, args)

            calculation_result = {
                "operation": func_name,
                "arguments": args,
                "result": float(result) if result.replace('.', '').replace('-', '').isdigit() else result
            }
            results.append(calculation_result)

        return {
            "original_request": request.expression,
            "previous_result": previous_result,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/statistics")
async def get_calculation_statistics():
    """Get calculation statistics from MCP server"""
    try:
        stats = await read_mcp_resource("calculation://statistics")
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.get("/summary")
async def get_calculation_summary():
    """Get calculation summary from MCP server"""
    try:
        summary = await read_mcp_resource("calculation://summary")
        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@app.post("/calculate/batch")
async def batch_calculations(expressions: list[str]):
    """Process multiple calculations in sequence, similar to notebook approach"""
    if not mistral_client:
        raise HTTPException(status_code=503, detail="Mistral AI not configured")

    results = []
    for expression in expressions:
        try:
            # Get previous result for context (like in the notebook)
            previous_result = await get_previous_result()

            # Generate new query with context
            message = generate_new_query(expression, previous_result)

            if message and message.tool_calls:
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name

                    if isinstance(tool_call.function.arguments, str):
                        args = json.loads(tool_call.function.arguments)
                    else:
                        args = tool_call.function.arguments

                    result = await execute_mcp_tool(func_name, args)

                    calculation_result = {
                        "expression": expression,
                        "operation": func_name,
                        "arguments": args,
                        "result": float(result) if result.replace('.', '').replace('-', '').isdigit() else result,
                        "previous_result": previous_result,
                        "success": True
                    }
                    results.append(calculation_result)
            else:
                results.append({
                    "expression": expression,
                    "error": "Could not interpret calculation",
                    "success": False
                })

        except Exception as e:
            results.append({
                "expression": expression,
                "error": str(e),
                "success": False
            })

    # Get final summary
    try:
        summary = await read_mcp_resource("calculation://summary")
    except:
        summary = None

    return {
        "batch_results": results,
        "final_summary": summary,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/clear_history")
async def clear_calculation_history():
    """Clear the calculation history on MCP server"""
    try:
        result = await execute_mcp_tool("clear_history", {})
        return {
            "message": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

@app.get("/last_result")
async def get_last_calculation_result():
    """Get the last calculation result from MCP server"""
    try:
        result = await get_previous_result()
        return {
            "last_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get last result: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MCP Calculator Client API",
        "version": "1.0.0",
        "description": "API for natural language calculations with MCP server and previous result context",
        "endpoints": {
            "calculate": "POST /calculate - Natural language calculation with context",
            "batch": "POST /calculate/batch - Multiple sequential calculations",
            "statistics": "GET /statistics - Get session statistics",
            "summary": "GET /summary - Get calculation summary",
            "clear_history": "POST /clear_history - Clear calculation history",
            "last_result": "GET /last_result - Get last calculation result"
        },
        "mcp_server": MCP_SERVER_URL,
        "mistral_configured": mistral_client is not None,
        "features": [
            "Previous result context awareness",
            "Sequential calculations with memory",
            "Natural language processing",
            "Session statistics and summaries"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with await get_mcp_session() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                stats = await read_mcp_resource("calculation://statistics")

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "mcp_server": "connected",
            "mistral_ai": "configured" if mistral_client else "not configured",
            "session_stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9232)