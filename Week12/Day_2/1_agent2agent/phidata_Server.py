import asyncio
import uuid
import uvicorn
from typing import Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import time

# FastA2A imports  
from fasta2a import FastA2A, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import (
    AgentCard, Skill, Message, TextPart, Artifact,
    TaskSendParams, TaskState
)

# PhiData imports
from phi.agent import Agent as PhiAgent
from phi.model.openai import OpenAIChat
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

env_vars_to_clear = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']
for var in env_vars_to_clear:
    if os.getenv(var):
        print(f"⚠️  Removing conflicting {var}")
        del os.environ[var]

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

class PhiDataWorker(Worker[list[Message]]):
    """FastA2A Worker that wraps a PhiData agent"""
    
    def __init__(self):
        self.phidata_agent = PhiAgent(
            model=OpenAIChat(id="gpt-4o-mini"),
            name="Strategy Planner",
            instructions=[
                "Create actionable strategic plans based on research data",
                "Provide clear, step-by-step recommendations",
                "Focus on practical implementation strategies"
            ],
            show_tool_calls=False
        )
    
    async def run_task(self, params: TaskSendParams) -> None:
        """Execute PhiData task via FastA2A protocol"""
        task = await self.storage.load_task(params['id'])
        if not task:
            return
            
        await self.storage.update_task(task['id'], state=TaskState.working)
        
        # Get conversation context
        context = await self.storage.load_context(task['context_id']) or []
        context.extend(task.get('history', []))
        
        # Extract user message
        user_message = ""
        for msg in context:
            if msg.get('role') == 'user':
                for part in msg.get('parts', []):
                    if part.get('kind') == 'text':
                        user_message += part.get('text', '')
        
        try:
            # Execute with PhiData
            response = self.phidata_agent.run(f"Create a strategic plan for: {user_message}")
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # Create response message
            response_message = Message(
                role='agent',
                parts=[TextPart(text=result_text, kind='text')],
                kind='message',
                message_id=str(uuid.uuid4())
            )
            
            # Create artifact
            artifact = Artifact(
                artifact_id=str(uuid.uuid4()),
                title="Strategic Plan",
                description="Strategic plan from PhiData agent",
                mime_type="text/plain",
                content=result_text
            )
            
            # Update task with results
            await self.storage.update_task(
                task['id'],
                state=TaskState.completed,
                new_messages=[response_message],
                new_artifacts=[artifact]
            )
            
            # Update context
            new_context = context + [response_message]
            await self.storage.save_context(task['context_id'], new_context)
            
        except Exception as e:
            # Handle errors
            error_message = Message(
                role='agent',
                parts=[TextPart(text=f"Error: {str(e)}", kind='text')],
                kind='message',
                message_id=str(uuid.uuid4())
            )
            
            await self.storage.update_task(
                task['id'],
                state=TaskState.failed,
                new_messages=[error_message]
            )
    
    # Required abstract method implementations
    async def build_artifacts(self, task_data: dict) -> list[Artifact]:
        """Build artifacts from task data"""
        artifacts = []
        if 'new_artifacts' in task_data:
            artifacts.extend(task_data['new_artifacts'])
        return artifacts
    
    async def build_message_history(self, task_data: dict) -> list[Message]:
        """Build message history from task data"""
        messages = []
        if 'new_messages' in task_data:
            messages.extend(task_data['new_messages'])
        return messages
    
    async def cancel_task(self, task_id: str) -> None:
        """Cancel a running task"""
        try:
            await self.storage.update_task(task_id, state=TaskState.cancelled)
            print(f"✅ Task {task_id} cancelled successfully")
        except Exception as e:
            print(f"❌ Error cancelling task {task_id}: {e}")

def create_phidata_server(port=9331):
    """Create FastA2A server for PhiData agent"""
    
    # Create FastA2A components
    storage = InMemoryStorage()
    broker = InMemoryBroker()
    worker = PhiDataWorker()
    
    # Simple task storage for tracking
    task_results = {}
    
    # Create FastA2A app
    try:
        app = FastA2A(storage=storage, broker=broker, worker=worker)
        print("✅ FastA2A app created successfully")
    except Exception as e:
        print(f"❌ Error creating FastA2A app: {e}")
        # Fallback: create FastAPI app manually
        app = FastAPI()
        print("✅ Using fallback FastAPI app")
    
    # Add agent discovery endpoint
    @app.get("/.well-known/agent.json")
    async def get_agent_info():
        return {
            "name": "PhiData Strategy Agent",
            "description": "Strategic planning specialist using PhiData framework",
            "url": f"http://localhost:{port}",
            "version": "1.0.0",
            "skills": [
                {
                    "id": "strategy",
                    "name": "Strategic Planning",
                    "description": "Create comprehensive strategic plans and roadmaps",
                    "categories": ["strategy", "planning", "analysis"],
                    "examples": ["Create go-to-market strategy", "Develop implementation roadmap"]
                }
            ]
        }
    
    # Enhanced message endpoint with immediate processing
    @app.post("/message/send")
    async def send_message(payload: dict):
        """Handle message and process immediately"""
        try:
            task_id = str(uuid.uuid4())
            message_content = ""
            
            # Extract message content
            if "message" in payload:
                msg = payload["message"]
                if "parts" in msg:
                    for part in msg["parts"]:
                        if part.get("kind") == "text":
                            message_content += part.get("text", "")
            
            print(f"🔄 Processing task {task_id}: {message_content[:50]}...")
            
            # Store initial task state
            task_results[task_id] = {
                "id": task_id,
                "state": "working",
                "message": message_content,
                "created_at": time.time()
            }
            
            # Process with PhiData in background
            async def process_task():
                try:
                    # Create and run PhiData agent
                    phidata_agent = PhiAgent(
                        model=OpenAIChat(id="gpt-4o-mini"),
                        name="Strategy Planner",
                        instructions=[
                            "Create actionable strategic plans based on research data",
                            "Provide clear, step-by-step recommendations",
                            "Focus on practical implementation strategies"
                        ],
                        show_tool_calls=False
                    )
                    
                    response = phidata_agent.run(f"Create a strategic plan for: {message_content}")
                    result_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Update task with results
                    task_results[task_id].update({
                        "state": "completed",
                        "response": result_text,
                        "completed_at": time.time(),
                        "messages": [
                            {
                                "role": "agent",
                                "parts": [{"kind": "text", "text": result_text}],
                                "kind": "message",
                                "message_id": str(uuid.uuid4())
                            }
                        ],
                        "artifacts": [
                            {
                                "artifact_id": str(uuid.uuid4()),
                                "title": "Strategic Plan",
                                "description": "Strategic plan from PhiData agent",
                                "mime_type": "text/plain",
                                "content": result_text
                            }
                        ]
                    })
                    
                    print(f"✅ Task {task_id} completed successfully")
                    
                except Exception as e:
                    print(f"❌ Task {task_id} failed: {e}")
                    task_results[task_id].update({
                        "state": "failed",
                        "error": str(e),
                        "failed_at": time.time()
                    })
            
            # Start background processing
            asyncio.create_task(process_task())
            
            return {
                "task_id": task_id,
                "status": "received",
                "message": "Task queued for processing"
            }
            
        except Exception as e:
            print(f"❌ Error in send_message: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    # Add task status endpoint
    @app.get("/task/{task_id}")
    async def get_task_status(task_id: str):
        """Get task status and results"""
        if task_id in task_results:
            return task_results[task_id]
        else:
            return {"error": "Task not found", "task_id": task_id}
    
    # Add task listing endpoint for debugging
    @app.get("/tasks")
    async def list_tasks():
        """List all tasks for debugging"""
        return {
            "tasks": list(task_results.keys()),
            "count": len(task_results)
        }
    
    return app

if __name__ == "__main__":
    print("🚀 Starting PhiData FastA2A Server...")
    phidata_app = create_phidata_server(9331)
    print("📡 PhiData server running on http://localhost:9331")
    print("🔍 Test discovery: curl http://localhost:9331/.well-known/agent.json")
    uvicorn.run(phidata_app, host="localhost", port=9331, log_level="info")