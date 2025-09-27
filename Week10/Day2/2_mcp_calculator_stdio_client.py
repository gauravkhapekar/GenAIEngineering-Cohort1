import asyncio
import json
import sys
import argparse
import os
from mistralai import Mistral
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()
server_mcp='2_mcp_calculator_stdio_server.py'
# Set your Mistral API key
if 'MISTRAL_KEY' not in os.environ:
    raise ValueError("MISTRAL_KEY environment variable is not set.")
mistral = Mistral(api_key=os.getenv('MISTRAL_KEY'))

async def run_mcp_client(message_text=None):
    try:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_mcp]
        )

        # Connect to the MCP server and initialize session
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize session
                await session.initialize()

                # Get available tools from the server
                tools_result = await session.list_tools()
                print(f"\n🧮 Calculator ready! Found {len(tools_result.tools)} tools")

                # List the tools
                for tool in tools_result.tools:
                    print(f"- {tool.name}: {tool.description}")

                # If message is provided, process it with Mistral
                if message_text:
                    await process_message_with_mistral(session, tools_result.tools, message_text)

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def convert_mcp_tools_to_mistral(mcp_tools):
    """Convert MCP tools to Mistral function format"""
    mistral_tools = []
    for tool in mcp_tools:
        mistral_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        }
        mistral_tools.append(mistral_tool)
    return mistral_tools

async def process_message_with_mistral(session, mcp_tools, message_text):
    """Process a message using Mistral and handle tool calls"""
    try:
        print(f"\n📝 Processing message: {message_text}")

        # Convert MCP tools to Mistral format
        mistral_tools = convert_mcp_tools_to_mistral(mcp_tools)

        # Call Mistral API with tools
        response = mistral.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful calculator assistant. Use the available tools to help with mathematical calculations."
                },
                {"role": "user", "content": message_text}
            ],
            tools=mistral_tools,
            tool_choice="auto"  # Let Mistral decide which tool to use
        )

        message = response.choices[0].message

        # Handle the response
        if hasattr(message, 'tool_calls') and message.tool_calls is not None:
            print(f"🤖 Mistral wants to use {len(message.tool_calls)} tool(s)")

            tool_results = []

            for tool_call in message.tool_calls:
                # Get function name from the tool call
                func_name = tool_call.function.name

                # Parse arguments (might be string or dict)
                if isinstance(tool_call.function.arguments, str):
                    args = json.loads(tool_call.function.arguments)
                else:
                    args = tool_call.function.arguments

                print(f"🔧 Calling tool: {func_name} with args: {args}")

                # Call the MCP tool and get the result
                result = await session.call_tool(func_name, args)
                result_text = result.content[0].text
                print(f"✅ Tool result: {result_text}")

                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": func_name,
                    "content": result_text
                })

            # Send tool results back to Mistral for final response
            final_response = mistral.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful calculator assistant."
                    },
                    {"role": "user", "content": message_text},
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls
                    },
                    *tool_results  # Tool results
                ]
            )

            final_message = final_response.choices[0].message.content
            print(f"\n🎯 Final answer: {final_message}")
        else:
            # No tool calls, just print the Mistral response
            content = message.content if hasattr(message, 'content') else str(message)
            print(f"🤖 Mistral response: {content}")

    except Exception as e:
        print(f"❌ Error processing message with Mistral: {e}")
        import traceback
        traceback.print_exc()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="MCP Client with Mistral integration for calculator operations")
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        default = 'What is 15 plus 27?',
        help="Message to process (e.g., 'What is 15 plus 27?' or 'Calculate 100 minus 45')"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model",
        default="mistral-small-latest",
        help="Mistral model to use (default: mistral-small-latest)"
    )
    return parser.parse_args()

async def interactive_mode(model="mistral-small-latest"):
    """Run the client in interactive mode with Mistral"""
    print("🎯 Interactive MCP Client with Mistral")
    print("Ask any math questions and I'll use the calculator tools to help!")
    print("Type 'quit' or 'exit' to stop\n")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_mcp]
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Get available tools
            tools_result = await session.list_tools()
            print(f"🧮 Connected! Available tools: {[tool.name for tool in tools_result.tools]}\n")

            while True:
                try:
                    message = input("💭 Ask me anything: ").strip()
                    if message.lower() in ['quit', 'exit', 'q']:
                        break

                    if message:
                        await process_message_with_mistral(session, tools_result.tools, message)
                        print()  # Add blank line for readability

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    break

def check_mistral_key():
    """Check if Mistral API key is set"""
    if not os.getenv('MISTRAL_KEY'):
        print("❌ Error: MISTRAL_KEY environment variable not set")
        print("Please set your Mistral API key:")
        print("export MISTRAL_KEY='your-api-key-here'")
        sys.exit(1)

def main():
    """Main function to run the MCP client"""
    check_mistral_key()
    args = parse_arguments()

    print("🚀 Starting MCP Client with Mistral integration...")

    try:
        if args.interactive:
            asyncio.run(interactive_mode(args.model))
        elif args.message:
            asyncio.run(run_mcp_client(args.message))
        else:
            # Default: just list tools
            asyncio.run(run_mcp_client())

    except KeyboardInterrupt:
        print("\n👋 MCP Client stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()