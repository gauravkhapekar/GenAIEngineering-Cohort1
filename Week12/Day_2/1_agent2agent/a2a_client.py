import asyncio
import uuid
from typing import Any
import threading
import uvicorn
import httpx
import time

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

# ============================================================================
# Enhanced A2A Communication Client
# ============================================================================

class A2AClient:
    """Enhanced client for A2A communication with task polling"""
    
    async def discover_agent(self, url: str):
        """Discover agent via A2A protocol"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                print(f"🔍 Trying to discover agent at: {url}/.well-known/agent.json")
                response = await client.get(f"{url}/.well-known/agent.json")
                print(f"📡 Response status: {response.status_code}")
                
                if response.status_code == 200:
                    agent_info = response.json()
                    print(f"✅ Agent discovered: {agent_info.get('name', 'Unknown')}")
                    return agent_info
                else:
                    print(f"❌ Discovery failed with status: {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
                    
            except httpx.ConnectError:
                print(f"❌ Connection failed to {url} - server may not be running")
                return None
            except Exception as e:
                print(f"❌ Discovery error: {e}")
                return None
    
    async def send_message_and_wait(self, url: str, message: str, context_id: str = None, max_wait_time: int = 60):
        """Send message and wait for completion with polling"""
        
        # Step 1: Send the message
        task_response = await self.send_message(url, message, context_id)
        if not task_response:
            return None
            
        task_id = task_response.get('task_id')
        if not task_id:
            print("❌ No task ID returned from server")
            return task_response
        
        print(f"⏳ Task {task_id} queued, waiting for completion...")
        
        # Step 2: Poll for task completion
        return await self.wait_for_task_completion(url, task_id, max_wait_time)
    
    async def send_message(self, url: str, message: str, context_id: str = None):
        """Send message to A2A agent"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                print(f"📤 Sending message to: {url}/message/send")
                
                # Create message payload
                payload = {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": message}],
                        "kind": "message",
                        "message_id": str(uuid.uuid4())
                    }
                }
                
                if context_id:
                    payload["context_id"] = context_id
                
                print(f"📋 Payload: {payload}")
                
                # Send to A2A endpoint
                response = await client.post(f"{url}/message/send", json=payload)
                print(f"📡 Message response status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(f"✅ Message sent successfully")
                        print(f"📋 Response data: {result}")
                        return result
                    except Exception as json_error:
                        print(f"❌ JSON decode error: {json_error}")
                        print(f"📄 Raw response: {response.text}")
                        return None
                else:
                    print(f"❌ Message failed with status: {response.status_code}")
                    print(f"📄 Response: {response.text}")
                    return None
                    
            except httpx.ConnectError:
                print(f"❌ Connection failed to {url}/message/send - server may not be running")
                return None
            except Exception as e:
                print(f"❌ Error sending message: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    async def wait_for_task_completion(self, url: str, task_id: str, max_wait_time: int = 60):
        """Poll for task completion and get results"""
        start_time = time.time()
        poll_interval = 2  # seconds
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.time() - start_time < max_wait_time:
                try:
                    # Check task status
                    status_response = await client.get(f"{url}/task/{task_id}")
                    
                    if status_response.status_code == 200:
                        task_data = status_response.json()
                        task_state = task_data.get('state', 'unknown')
                        
                        print(f"📊 Task {task_id} status: {task_state}")
                        
                        if task_state == 'completed':
                            print(f"✅ Task {task_id} completed!")
                            
                            # Extract results
                            messages = task_data.get('messages', [])
                            artifacts = task_data.get('artifacts', [])
                            
                            # Find agent response
                            agent_response = None
                            for msg in messages:
                                if msg.get('role') == 'agent':
                                    for part in msg.get('parts', []):
                                        if part.get('kind') == 'text':
                                            agent_response = part.get('text', '')
                                            break
                                    if agent_response:
                                        break
                            
                            return {
                                'task_id': task_id,
                                'status': 'completed',
                                'response': agent_response,
                                'artifacts': artifacts,
                                'full_data': task_data
                            }
                        
                        elif task_state == 'failed':
                            print(f"❌ Task {task_id} failed")
                            return {
                                'task_id': task_id,
                                'status': 'failed',
                                'error': task_data.get('error', 'Unknown error')
                            }
                        
                        elif task_state in ['working', 'pending']:
                            print(f"⏳ Task {task_id} still {task_state}...")
                            await asyncio.sleep(poll_interval)
                            continue
                        
                    else:
                        print(f"❌ Error checking task status: {status_response.status_code}")
                        break
                        
                except Exception as e:
                    print(f"❌ Error polling task {task_id}: {e}")
                    break
            
            # Timeout
            print(f"⏰ Task {task_id} timed out after {max_wait_time} seconds")
            return {
                'task_id': task_id,
                'status': 'timeout',
                'error': f'Task did not complete within {max_wait_time} seconds'
            }

# ============================================================================
# Enhanced Demo: FastA2A Agent Communication
# ============================================================================

async def main():
    print("🚀 FastA2A Demo: CrewAI ↔ PhiData Communication")
    print("=" * 55)
    
    # Initialize A2A client
    client = A2AClient()
    
    # Step 1: Agent Discovery
    print("\n🔍 Step 1: Agent Discovery via FastA2A")
    crewai_info = await client.discover_agent("http://localhost:9321")
    phidata_info = await client.discover_agent("http://localhost:9331")
    
    if crewai_info:
        print(f"✅ Discovered: {crewai_info['name']}")
        print(f"   Skills: {[s['name'] for s in crewai_info.get('skills', [])]}")
    else:
        print("❌ CrewAI agent not discovered - check if server is running on port 9321")
        return
    
    if phidata_info:
        print(f"✅ Discovered: {phidata_info['name']}")
        print(f"   Skills: {[s['name'] for s in phidata_info.get('skills', [])]}")
    else:
        print("❌ PhiData agent not discovered - check if server is running on port 9331")
        return
    
    # Step 2: Send research request to CrewAI and wait for completion
    print("\n📈 Step 2: CrewAI Research via FastA2A")
    print("📤 Sending: 'Research the current state of AI agent market adoption'")
    
    research_result = await client.send_message_and_wait(
        "http://localhost:9321",
        "Research the current state of AI agent market adoption",
        max_wait_time=120  # 2 minutes timeout
    )
    
    if research_result and research_result.get('status') == 'completed':
        print("✅ Research completed!")
        print(f"📋 Task ID: {research_result.get('task_id')}")
        print(f"🔍 Research Results:")
        print("=" * 50)
        print(research_result.get('response', 'No response content'))
        print("=" * 50)
        
        # Show artifacts if any
        artifacts = research_result.get('artifacts', [])
        if artifacts:
            print(f"📎 Generated {len(artifacts)} artifact(s)")
            for i, artifact in enumerate(artifacts, 1):
                print(f"   {i}. {artifact.get('title', 'Untitled')}")
    else:
        print("❌ Research task failed or timed out")
        if research_result:
            print(f"   Status: {research_result.get('status')}")
            print(f"   Error: {research_result.get('error', 'Unknown')}")
    
    # Step 3: Send planning request to PhiData and wait for completion
    print("\n🎯 Step 3: PhiData Planning via FastA2A")
    print("📤 Sending: 'Create a go-to-market strategy for AI agent platforms'")
    
    planning_result = await client.send_message_and_wait(
        "http://localhost:9331",
        "Create a go-to-market strategy for AI agent platforms based on the research above",
        max_wait_time=120  # 2 minutes timeout
    )
    
    if planning_result and planning_result.get('status') == 'completed':
        print("✅ Strategic planning completed!")
        print(f"📋 Task ID: {planning_result.get('task_id')}")
        print(f"🎯 Strategic Plan:")
        print("=" * 50)
        print(planning_result.get('response', 'No response content'))
        print("=" * 50)
        
        # Show artifacts if any
        artifacts = planning_result.get('artifacts', [])
        if artifacts:
            print(f"📎 Generated {len(artifacts)} artifact(s)")
            for i, artifact in enumerate(artifacts, 1):
                print(f"   {i}. {artifact.get('title', 'Untitled')}")
    else:
        print("❌ Planning task failed or timed out")
        if planning_result:
            print(f"   Status: {planning_result.get('status')}")
            print(f"   Error: {planning_result.get('error', 'Unknown')}")
    
    print("\n✅ FastA2A Communication Demo Complete!")
    print("\n🎉 Demonstration Summary:")
    print("   • Agent discovery via standardized endpoints")
    print("   • Asynchronous task submission")
    print("   • Real-time task status polling")
    print("   • Complete response retrieval")
    print("   • Artifact management")
    print("   • Cross-framework communication (CrewAI ↔ PhiData)")

# Add connectivity test function
async def test_connectivity():
    """Test basic connectivity to servers"""
    print("🔧 Testing server connectivity...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        for port in [9321, 9331]:
            try:
                response = await client.get(f"http://localhost:{port}/.well-known/agent.json")
                print(f"✅ Port {port}: {response.status_code} - {response.json().get('name', 'Unknown')}")
            except httpx.ConnectError:
                print(f"❌ Port {port}: Connection refused - server not running")
            except Exception as e:
                print(f"❌ Port {port}: {e}")

if __name__ == "__main__":
    print("📦 Setup: pip install fasta2a crewai phidata uvicorn httpx")
    print("🔑 Required: OPEN_ROUTER_KEY in .env file")
    print()
    
    try:
        # Test connectivity first
        asyncio.run(test_connectivity())
        print()
        
        # Run main demo
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Ensure FastA2A and dependencies are installed")
        import traceback
        traceback.print_exc()