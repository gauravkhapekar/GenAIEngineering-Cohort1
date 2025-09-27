import streamlit as st
import asyncio
import subprocess
import tempfile
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import time
from pathlib import Path
import threading
from datetime import datetime
import zipfile

# LangGraph-style state management
class NodeState(Enum):
    PLANNING = "planning"
    CODING = "coding" 
    TESTING = "testing"
    DEBUGGING = "debugging"
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class LogEntry:
    timestamp: str
    node: str
    level: str  # INFO, DEBUG, WARNING, ERROR
    message: str
    details: Optional[str] = None

@dataclass
class CodeFile:
    filename: str
    content: str
    description: str = ""

@dataclass
class AgentState:
    feature_request: str = ""
    plan: str = ""
    planned_files: List[str] = None  # List of files mentioned in the plan
    code_files: List[CodeFile] = None  # Generated code files
    main_code: str = ""  # Primary/main application code
    test_code: str = ""
    test_results: str = ""
    errors: List[str] = None
    current_node: NodeState = NodeState.PLANNING
    iteration: int = 0
    max_iterations: int = 5
    logs: List[LogEntry] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.logs is None:
            self.logs = []
        if self.planned_files is None:
            self.planned_files = []
        if self.code_files is None:
            self.code_files = []
    
    def add_log(self, node: str, level: str, message: str, details: str = None):
        """Add a new log entry with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = LogEntry(timestamp, node, level, message, details)
        self.logs.append(log_entry)
        return log_entry

class OpenRouterClient:
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-sonnet"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate(self, messages: List[Dict], temperature: float = 0.7) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"

class CodeAgent:
    def __init__(self, openrouter_client: OpenRouterClient):
        self.client = openrouter_client
        self.temp_dir = tempfile.mkdtemp()
    
    def planning_node(self, state: AgentState) -> AgentState:
        """Generate a detailed plan for the feature request"""
        state.add_log("PLANNING", "INFO", "Starting planning phase...")
        
        messages = [
            {"role": "system", "content": """You are a senior software architect. Create a detailed implementation plan for the given feature request.

Your plan should include:
1. **Project Structure**: List ALL files that need to be created with their exact filenames
2. **File Descriptions**: Brief description of what each file should contain
3. **Key Components**: Main classes, functions, and their responsibilities  
4. **External Dependencies**: Required packages and imports
5. **Implementation Strategy**: Step-by-step approach
6. **Testing Strategy**: How to test the implementation

Be very specific about the file structure. Use this format for the files section:

FILES TO CREATE:
- main.py: Main application entry point
- utils.py: Utility functions and helpers  
- config.py: Configuration settings
- requirements.txt: Python dependencies

Make sure to list every file that will be needed for a complete implementation."""},
            {"role": "user", "content": f"Feature Request: {state.feature_request}"}
        ]
        
        plan = self.client.generate(messages, temperature=0.3)
        state.plan = plan
        
        # Extract planned files from the plan
        state.planned_files = self._extract_planned_files(plan)
        state.add_log("PLANNING", "INFO", f"Identified {len(state.planned_files)} files to create: {', '.join(state.planned_files)}")
        
        state.current_node = NodeState.CODING
        state.add_log("PLANNING", "SUCCESS", "Planning phase completed successfully")
        return state
    
    def _extract_planned_files(self, plan: str) -> List[str]:
        """Extract the list of files that need to be created from the plan"""
        planned_files = []
        lines = plan.split('\n')
        
        # Look for various patterns that indicate file listings
        file_indicators = [
            'FILES TO CREATE:',
            'File Structure:',
            'Files needed:',
            'Project files:',
            'Required files:'
        ]
        
        in_file_section = False
        for line in lines:
            line = line.strip()
            
            # Check if we're entering a file section
            if any(indicator.lower() in line.lower() for indicator in file_indicators):
                in_file_section = True
                continue
            
            # Stop if we hit a new section
            if in_file_section and line and not line.startswith(('-', '*', '•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if line.isupper() or line.endswith(':'):
                    break
            
            # Extract filenames
            if in_file_section and line:
                # Look for patterns like "- filename.py:" or "* filename.py -"
                import re
                file_patterns = [
                    r'[-*•]\s*([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z]{2,4})',  # - filename.ext
                    r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z]{2,4}):',         # filename.ext:
                    r'`([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z]{2,4})`',        # `filename.ext`
                    r'(\w+\.py|\w+\.txt|\w+\.json|\w+\.md|\w+\.yml|\w+\.yaml|\w+\.html|\w+\.css|\w+\.js)'  # common extensions
                ]
                
                for pattern in file_patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        if match not in planned_files and '.' in match:
                            planned_files.append(match)
        
        # If no files found in structured format, look for common files mentioned anywhere
        if not planned_files:
            import re
            all_files = re.findall(r'\b(\w+\.[a-zA-Z]{2,4})\b', plan)
            # Filter for reasonable filenames
            for filename in all_files:
                if (filename.endswith(('.py', '.txt', '.json', '.md', '.yml', '.yaml', '.html', '.css', '.js')) 
                    and len(filename) > 3 and filename not in planned_files):
                    planned_files.append(filename)
        
        # Ensure we have at least main.py if nothing else is found
        if not planned_files:
            planned_files = ['main.py']
        
        return planned_files[:10]  # Limit to reasonable number of files
    
    def coding_node(self, state: AgentState) -> AgentState:
        """Generate code for all planned files, one by one with UI updates"""
        state.add_log("CODING", "INFO", f"Starting code generation for {len(state.planned_files)} files...")
        
        # If this is the first time entering coding, clear previous code files
        if not hasattr(state, '_coding_file_index'):
            state.code_files = []
            state._coding_file_index = 0
        
        # Check if we've completed all files
        if state._coding_file_index >= len(state.planned_files):
            state.current_node = NodeState.TESTING
            state.add_log("CODING", "SUCCESS", f"Code generation completed for all {len(state.code_files)} files")
            # Clean up the temporary attribute
            delattr(state, '_coding_file_index')
            return state
        
        # Generate code for the current file
        filename = state.planned_files[state._coding_file_index]
        current_file_num = state._coding_file_index + 1
        total_files = len(state.planned_files)
        
        state.add_log("CODING", "INFO", f"Generating code for {filename} ({current_file_num}/{total_files})")
        
        messages = [
            {"role": "system", "content": f"""You are an expert Python developer. Generate complete, production-ready code for the specific file: {filename}

Context: You are implementing this as part of a larger project. Here's the overall plan:

{state.plan}

Requirements for {filename}:
- Write complete, functional code for this specific file
- Include proper imports and dependencies
- Add docstrings and comments
- Handle errors appropriately
- Make sure this file works as part of the overall project
- If this is the main application file, ensure it can be run directly
- If this is a utility/helper file, ensure it provides the needed functions

IMPORTANT: 
- Only generate code for {filename}, not other files
- Make sure the code is complete and functional
- Include all necessary imports
- If this file depends on other files in the project, assume they exist and import them properly

Format your response as:
```python
# Complete code for {filename}
```"""},
            {"role": "user", "content": f"""
Generate the complete code for: {filename}

Feature Request: {state.feature_request}

Project Plan: {state.plan}

Other files in project: {[f for f in state.planned_files if f != filename]}

{f"Previous errors to consider: {'; '.join(state.errors)}" if state.errors else ""}
"""}
        ]
        
        response = self.client.generate(messages, temperature=0.5)
        
        # Extract code from markdown
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            code = response.split("```")[1].split("```")[0].strip()
        else:
            code = response.strip()
        
        # Create code file object
        code_file = CodeFile(
            filename=filename,
            content=code,
            description=f"Generated code for {filename}"
        )
        state.code_files.append(code_file)
        
        # Set main code to the primary file (usually main.py or the first file)
        if filename == 'main.py' or (not state.main_code and state._coding_file_index == 0):
            state.main_code = code
        
        state.add_log("CODING", "SUCCESS", f"Generated {len(code.split())} lines for {filename}")
        
        # Move to next file
        state._coding_file_index += 1
        
        # Stay in coding phase to generate the next file
        # The node will be called again for the next iteration
        return state
    
    def testing_node(self, state: AgentState) -> AgentState:
        """Generate and run comprehensive tests for all code files"""
        if not state.test_code:
            state.add_log("TESTING", "INFO", "Generating comprehensive test suite...")
            
            # Create a summary of all generated code for test generation
            code_summary = "GENERATED CODE FILES:\n\n"
            for code_file in state.code_files:
                code_summary += f"=== {code_file.filename} ===\n"
                code_summary += code_file.content + "\n\n"
            
            messages = [
                {"role": "system", "content": """You are a testing expert. Write comprehensive unit tests for the given codebase.

Requirements:
- Use pytest framework
- Test all major functions and classes across all files
- Cover main functionality and edge cases
- Include integration tests if multiple files work together
- Test error conditions and input validation
- Mock external dependencies if needed
- Make tests independent and repeatable
- Import all necessary modules from the codebase

The codebase consists of multiple files - make sure to test functionality across all of them.

Format your response as:
```python
# Complete test suite for the entire codebase
```"""},
                {"role": "user", "content": f"""
Generate comprehensive tests for this codebase:

{code_summary}

Feature Request: {state.feature_request}

Files to test: {[cf.filename for cf in state.code_files]}

Make sure to import and test functions/classes from all the generated files.
"""}
            ]
            
            response = self.client.generate(messages, temperature=0.3)
            
            # Extract test code
            if "```python" in response:
                test_code = response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                test_code = response.split("```")[1].split("```")[0].strip()
            else:
                test_code = response.strip()
                
            state.test_code = test_code
            state.add_log("TESTING", "SUCCESS", "Comprehensive test suite generated")
        else:
            state.add_log("TESTING", "INFO", "Using existing test code for validation")
        
        state.add_log("TESTING", "INFO", "Running unit tests on complete codebase...")
        
        # Run the tests against all generated code
        test_results = self._run_comprehensive_tests(state.code_files, state.test_code)
        state.test_results = test_results
        
        # Analyze test results
        if "FAILED" in test_results or "ERROR" in test_results or "Error" in test_results:
            errors = self._extract_errors(test_results)
            state.errors.extend(errors)
            
            failed_count = test_results.count("FAILED")
            error_count = test_results.count("ERROR")
            
            state.add_log("TESTING", "ERROR", f"Tests failed: {failed_count} failures, {error_count} errors")
            
            for i, error in enumerate(errors[:3], 1):
                state.add_log("TESTING", "DEBUG", f"Error {i}: {error}")
            
            state.current_node = NodeState.DEBUGGING
        elif "passed" in test_results.lower() or "ok" in test_results.lower():
            state.add_log("TESTING", "SUCCESS", "All tests passed successfully!")
            state.current_node = NodeState.COMPLETE
        else:
            if test_results.strip():
                state.add_log("TESTING", "WARNING", "Test results unclear - assuming failure for safety")
                state.errors.append("Unclear test results: " + test_results[:200])
                state.current_node = NodeState.DEBUGGING
            else:
                state.add_log("TESTING", "ERROR", "No test results received")
                state.errors.append("No test output received")
                state.current_node = NodeState.DEBUGGING
            
        return state
    
    def debugging_node(self, state: AgentState) -> AgentState:
        """Debug and fix issues across all code files"""
        state.iteration += 1
        
        state.add_log("DEBUGGING", "INFO", f"Starting debug iteration {state.iteration}/{state.max_iterations}")
        
        if state.iteration >= state.max_iterations:
            state.current_node = NodeState.FAILED
            state.add_log("DEBUGGING", "ERROR", "Maximum iterations reached - debugging failed")
            return state
        
        if state.errors:
            state.add_log("DEBUGGING", "INFO", f"Analyzing {len(state.errors)} identified errors")
            
            # Log each error clearly for user visibility
            for i, error in enumerate(state.errors, 1):
                state.add_log("DEBUGGING", "ERROR", f"Issue {i}: {error}")
        else:
            state.add_log("DEBUGGING", "WARNING", "No specific errors identified, analyzing test output")
        
        # Create comprehensive error analysis
        error_analysis = self._analyze_errors(state.test_results, state.errors)
        state.add_log("DEBUGGING", "INFO", f"Error analysis: {error_analysis[:200]}...")
        
        # Create comprehensive context for debugging
        current_code_summary = "CURRENT CODEBASE:\n\n"
        for code_file in state.code_files:
            current_code_summary += f"=== {code_file.filename} ===\n"
            current_code_summary += code_file.content + "\n\n"
        
        messages = [
            {"role": "system", "content": """You are a debugging expert. Analyze the multi-file codebase, test results, and errors to fix all issues.

CRITICAL REQUIREMENTS:
- Identify the ROOT CAUSE of each error
- Explain what's wrong before providing fixes
- Fix bugs while maintaining functionality
- Ensure all files work together properly
- Handle cross-file dependencies correctly
- Fix imports, syntax errors, logic errors, and runtime errors

For your response, FIRST explain what you found wrong, then provide the corrected code.

Format your response like this:

ANALYSIS:
- Error 1: [Explain what's wrong and why]
- Error 2: [Explain what's wrong and why]
- Root cause: [Main issue causing problems]

FIXES:
```python
# === FILENAME: main.py ===
# corrected code for main.py

# === FILENAME: utils.py ===  
# corrected code for utils.py

# (continue for all files that need fixes...)
```"""},
            {"role": "user", "content": f"""
DEBUG REQUEST:
Original Feature: {state.feature_request}

CURRENT CODEBASE:
{current_code_summary}

TEST CODE:
{state.test_code}

TEST RESULTS:
{state.test_results}

SPECIFIC ERRORS FOUND:
{chr(10).join([f"- {error}" for error in state.errors])}

ERROR ANALYSIS:
{error_analysis}

Please analyze what's wrong and provide corrected code for ALL files that need fixes.
"""}
        ]
        
        state.add_log("DEBUGGING", "INFO", "Analyzing errors and generating fixes...")
        response = self.client.generate(messages, temperature=0.4)
        
        # Extract the analysis and fixes from the response
        analysis, fixes = self._parse_debug_response(response)
        
        if analysis:
            state.add_log("DEBUGGING", "INFO", f"Debug Analysis: {analysis}")
        
        # Parse the multi-file response
        updated_files = self._parse_multi_file_response(fixes if fixes else response, state.code_files)
        
        if updated_files:
            # Log what was fixed
            state.add_log("DEBUGGING", "SUCCESS", f"Applied fixes to {len(updated_files)} files")
            
            state.code_files = updated_files
            # Update main code if main.py was updated
            main_file = next((cf for cf in state.code_files if cf.filename == 'main.py'), None)
            if main_file:
                state.main_code = main_file.content
            elif state.code_files:
                state.main_code = state.code_files[0].content
            
            state.errors = []  # Clear previous errors
            state.current_node = NodeState.TESTING
            
            state.add_log("DEBUGGING", "SUCCESS", f"Code fixes applied - returning to testing (iteration {state.iteration})")
        else:
            state.add_log("DEBUGGING", "ERROR", "Failed to generate valid fixed code")
            state.current_node = NodeState.FAILED
            
        return state
    
    def _analyze_errors(self, test_results: str, errors: List[str]) -> str:
        """Analyze errors to provide better context"""
        analysis = []
        
        if not test_results and not errors:
            return "No test results or errors available for analysis"
        
        # Analyze test results for common patterns
        if "ImportError" in test_results or "ModuleNotFoundError" in test_results:
            analysis.append("Import/module issues detected - likely missing dependencies or incorrect imports")
        
        if "SyntaxError" in test_results or "IndentationError" in test_results:
            analysis.append("Syntax/indentation errors found - code structure issues")
        
        if "NameError" in test_results:
            analysis.append("Variable/function name errors - undefined variables or functions")
        
        if "AttributeError" in test_results:
            analysis.append("Attribute errors - accessing non-existent methods or properties")
        
        if "TypeError" in test_results:
            analysis.append("Type errors - incorrect argument types or function calls")
        
        if "AssertionError" in test_results:
            analysis.append("Logic errors - code doesn't behave as expected")
        
        if "FAILED" in test_results:
            failed_count = test_results.count("FAILED")
            analysis.append(f"{failed_count} test(s) failed - functional issues in implementation")
        
        # Analyze specific errors
        for error in errors:
            if "import" in error.lower():
                analysis.append("Import path or dependency issue")
            elif "syntax" in error.lower():
                analysis.append("Code syntax problem")
            elif "name" in error.lower():
                analysis.append("Variable or function naming issue")
        
        return "; ".join(analysis) if analysis else "Generic errors detected in code execution"
    
    def _parse_debug_response(self, response: str) -> tuple:
        """Parse debug response to extract analysis and fixes separately"""
        analysis_section = ""
        fixes_section = ""
        
        # Look for ANALYSIS section
        if "ANALYSIS:" in response:
            parts = response.split("ANALYSIS:", 1)
            if len(parts) > 1:
                remaining = parts[1]
                if "FIXES:" in remaining:
                    analysis_section = remaining.split("FIXES:", 1)[0].strip()
                    fixes_section = remaining.split("FIXES:", 1)[1].strip()
                else:
                    analysis_section = remaining.strip()
        
        # If no structured format, try to extract analysis from beginning
        if not analysis_section and not fixes_section:
            if "```python" in response:
                parts = response.split("```python", 1)
                analysis_section = parts[0].strip()
                fixes_section = "```python" + parts[1]
            else:
                analysis_section = response[:300] + "..." if len(response) > 300 else response
                fixes_section = response
        
        return analysis_section, fixes_section
    
    def _parse_multi_file_response(self, response: str, existing_files: List[CodeFile]) -> List[CodeFile]:
        """Parse a multi-file response and return updated CodeFile objects"""
        updated_files = []
        
        # Look for file separators
        import re
        
        # Split by file markers
        file_sections = re.split(r'#\s*===\s*FILENAME:\s*([^=\n]+)\s*===', response)
        
        if len(file_sections) > 1:
            # Paired sections: [before_first_file, filename1, content1, filename2, content2, ...]
            for i in range(1, len(file_sections), 2):
                if i + 1 < len(file_sections):
                    filename = file_sections[i].strip()
                    content = file_sections[i + 1].strip()
                    
                    # Clean up the content
                    if content.startswith('```python'):
                        content = content[9:].strip()
                    if content.endswith('```'):
                        content = content[:-3].strip()
                    
                    updated_files.append(CodeFile(
                        filename=filename,
                        content=content,
                        description=f"Debugged code for {filename}"
                    ))
        else:
            # Fallback: try to extract single file or update existing files
            if "```python" in response:
                code = response.split("```python")[1].split("```")[0].strip()
            elif "```" in response:
                code = response.split("```")[1].split("```")[0].strip()
            else:
                code = response.strip()
            
            if code and len(code) > 10:
                # If we have existing files, update the first/main one
                if existing_files:
                    main_file = existing_files[0]
                    updated_files.append(CodeFile(
                        filename=main_file.filename,
                        content=code,
                        description=f"Debugged code for {main_file.filename}"
                    ))
                    # Keep other files unchanged
                    updated_files.extend(existing_files[1:])
                else:
                    updated_files.append(CodeFile(
                        filename="main.py",
                        content=code,
                        description="Debugged main code"
                    ))
        
        return updated_files
    
    def _run_comprehensive_tests(self, code_files: List[CodeFile], test_code: str) -> str:
        """Run tests against all generated code files"""
        try:
            # Create all code files in temp directory
            for code_file in code_files:
                file_path = os.path.join(self.temp_dir, code_file.filename)
                with open(file_path, "w") as f:
                    f.write(code_file.content)
            
            # Create test file
            test_file = os.path.join(self.temp_dir, "test_complete.py")
            with open(test_file, "w") as f:
                f.write(test_code)
            
            # Run pytest
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v"],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nReturn Code: {result.returncode}"
            
        except subprocess.TimeoutExpired:
            return "Error: Test execution timed out"
        except Exception as e:
            return f"Error running tests: {str(e)}"
    
    def _run_tests(self, code: str, test_code: str) -> str:
        """Legacy method - kept for backward compatibility"""
        return self._run_comprehensive_tests([CodeFile("main.py", code)], test_code)
    
    def _extract_errors(self, test_results: str) -> List[str]:
        """Extract specific error messages from test results"""
        errors = []
        lines = test_results.split('\n')
        
        error_patterns = [
            "FAILED", "ERROR", "AssertionError", "ImportError", "ModuleNotFoundError",
            "NameError", "SyntaxError", "IndentationError", "TypeError", "ValueError", "AttributeError"
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in error_patterns:
                if pattern in line:
                    error_context = line
                    if i + 1 < len(lines) and lines[i + 1].strip():
                        error_context += " | " + lines[i + 1].strip()
                    errors.append(error_context)
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_errors = []
        for error in errors:
            if error not in seen:
                seen.add(error)
                unique_errors.append(error)
        
        return unique_errors[:5]

def get_phase_status(state: AgentState, phase: NodeState) -> str:
    """Get the status of a phase"""
    phase_order = [NodeState.PLANNING, NodeState.CODING, NodeState.TESTING, NodeState.DEBUGGING, NodeState.COMPLETE, NodeState.FAILED]
    
    try:
        current_index = phase_order.index(state.current_node)
        phase_index = phase_order.index(phase)
        
        # Special handling for coding phase
        if phase == NodeState.CODING:
            if state.current_node == NodeState.CODING:
                return "running"
            elif hasattr(state, '_coding_file_index') or state.code_files:
                return "success"  # Coding has started/completed
            elif current_index > phase_index:
                return "success"
            else:
                return "pending"
        
        if phase_index < current_index:
            return "success"
        elif phase_index == current_index:
            return "running"
        else:
            return "pending"
    except ValueError:
        return "pending"

def create_project_zip(state: AgentState) -> bytes:
    """Create a zip file containing all generated code files"""
    import io
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all code files
        for code_file in state.code_files:
            zip_file.writestr(code_file.filename, code_file.content)
        
        # Add test file if exists
        if state.test_code:
            zip_file.writestr("tests.py", state.test_code)
        
        # Add project plan as README
        if state.plan:
            readme_content = f"# Project Plan\n\n{state.plan}\n\n## Generated Files\n\n"
            for code_file in state.code_files:
                readme_content += f"- **{code_file.filename}**: {code_file.description}\n"
            zip_file.writestr("README.md", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def update_phase_ui(state: AgentState, unique_suffix=""):
    """Update the UI for all phases"""
    
    st.markdown("## 🔄 Agent Progress")
    
    # Planning Phase
    planning_status = get_phase_status(state, NodeState.PLANNING)
    
    st.markdown(f"""
    <div class="phase-container">
        <div class="phase-header">
            🎯 Planning Phase
            <span class="status-badge status-{planning_status}">
                {'✅ Complete' if planning_status == 'success' else '🔄 Running' if planning_status == 'running' else '⏳ Pending'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if state.plan:
        with st.expander("📋 View Implementation Plan", expanded=(planning_status == "running")):
            st.markdown("### 📊 Generated Plan")
            st.markdown(state.plan)
            
            # Plan metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Words", len(state.plan.split()))
            with col2:
                st.metric("Sections", len(state.plan.split('\n\n')))
            with col3:
                st.metric("Characters", len(state.plan))
            with col4:
                st.metric("Planned Files", len(state.planned_files))
            
            if state.planned_files:
                st.markdown("### 📁 Files to Generate")
                for i, filename in enumerate(state.planned_files, 1):
                    st.markdown(f"{i}. `{filename}`")
    
    # Coding Phase  
    coding_status = get_phase_status(state, NodeState.CODING)
    
    st.markdown(f"""
    <div class="phase-container">
        <div class="phase-header">
            💻 Coding Phase
            <span class="status-badge status-{coding_status}">
                {'✅ Complete' if coding_status == 'success' else '🔄 Running' if coding_status == 'running' else '⏳ Pending'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if state.code_files or (hasattr(state, '_coding_file_index') and coding_status == "running"):
        with st.expander("💻 View Generated Code Files", expanded=(coding_status == "running")):
            # Show progress if currently coding
            if hasattr(state, '_coding_file_index') and coding_status == "running":
                current_file = state._coding_file_index
                total_files = len(state.planned_files)
                progress = current_file / total_files if total_files > 0 else 0
                
                st.markdown(f"### 🔄 Generating Files... ({current_file}/{total_files})")
                st.progress(progress)
                
                if current_file > 0:
                    st.success(f"✅ Completed: {', '.join(cf.filename for cf in state.code_files)}")
                
                if current_file < total_files:
                    remaining_files = state.planned_files[current_file:]
                    st.info(f"⏳ Remaining: {', '.join(remaining_files)}")
            
            if state.code_files:
                st.markdown(f"### 🔨 Generated Code ({len(state.code_files)} files)")
                
                # Use session state to maintain file selection across reruns
                if f"selected_file_index" not in st.session_state:
                    st.session_state.selected_file_index = 0
                
                # Create file tabs instead of dropdown to avoid rerun issues
                if len(state.code_files) == 1:
                    # If only one file, just show it
                    selected_code_file = state.code_files[0]
                    st.markdown(f"#### 📄 {selected_code_file.filename}")
                    st.code(selected_code_file.content, language="python", line_numbers=True)
                else:
                    # Multiple files - use tabs
                    file_tabs = st.tabs([f"📄 {cf.filename}" for cf in state.code_files])
                    
                    for i, (tab, code_file) in enumerate(zip(file_tabs, state.code_files)):
                        with tab:
                            line_count = len([line for line in code_file.content.split('\n') if line.strip()])
                            st.markdown(f"**{code_file.filename}** ({line_count} lines)")
                            st.markdown(f"*{code_file.description}*")
                            st.code(code_file.content, language="python", line_numbers=True)
                
                # Overall code metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_lines = sum(len([line for line in cf.content.split('\n') 
                                     if line.strip() and not line.strip().startswith('#')]) 
                                for cf in state.code_files)
                total_functions = sum(cf.content.count("def ") for cf in state.code_files)
                total_classes = sum(cf.content.count("class ") for cf in state.code_files)
                total_imports = sum(len([line for line in cf.content.split('\n') 
                                       if line.strip().startswith(('import', 'from'))]) 
                                  for cf in state.code_files)
                
                with col1:
                    st.metric("Total Lines", total_lines)
                with col2:
                    st.metric("Total Functions", total_functions)
                with col3:
                    st.metric("Total Classes", total_classes)
                with col4:
                    st.metric("Total Imports", total_imports)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    if state.main_code:
                        st.download_button(
                            label="📥 Download Main File",
                            data=state.main_code,
                            file_name="main.py",
                            mime="text/plain",
                            key=f"download_main_{unique_suffix}"
                        )
                
                with col2:
                    if len(state.code_files) > 1:
                        zip_data = create_project_zip(state)
                        st.download_button(
                            label="📦 Download Complete Project",
                            data=zip_data,
                            file_name="generated_project.zip",
                            mime="application/zip",
                            key=f"download_project_{unique_suffix}"
                        )
    
    # Testing Phase
    testing_status = get_phase_status(state, NodeState.TESTING)
    
    st.markdown(f"""
    <div class="phase-container">
        <div class="phase-header">
            🧪 Testing Phase
            <span class="status-badge status-{testing_status}">
                {'✅ Complete' if testing_status == 'success' else '🔄 Running' if testing_status == 'running' else '⏳ Pending'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if state.test_code:
        with st.expander("🧪 View Test Suite", expanded=(testing_status == "running")):
            st.markdown("### 🔬 Test Implementation")
            st.code(state.test_code, language="python", line_numbers=True)
            
            # Test metrics
            col1, col2, col3 = st.columns(3)
            test_functions = state.test_code.count("def test_")
            assertions = state.test_code.count("assert")
            test_lines = len([line for line in state.test_code.split('\n') if line.strip()])
            
            with col1:
                st.metric("Test Functions", test_functions)
            with col2:
                st.metric("Assertions", assertions)
            with col3:
                st.metric("Test Lines", test_lines)
            
            if state.test_results:
                st.markdown("### 📊 Test Results")
                
                # Parse test results for better display
                if "PASSED" in state.test_results and "FAILED" not in state.test_results:
                    st.success("🎉 All tests passed!")
                elif "FAILED" in state.test_results:
                    st.error("❌ Some tests failed")
                
                st.code(state.test_results, language="text")
            
            # Download test code
            st.download_button(
                label="📥 Download Tests",
                data=state.test_code,
                file_name="test_code.py",
                mime="text/plain",
                key=f"download_tests_{unique_suffix}"
            )
    
    # Debugging Phase (only show if there were iterations)
    if state.iteration > 0:
        debugging_status = get_phase_status(state, NodeState.DEBUGGING)
        
        st.markdown(f"""
        <div class="phase-container">
            <div class="phase-header">
                🐛 Debugging Phase
                <span class="status-badge status-{debugging_status}">
                    {'✅ Complete' if debugging_status == 'success' else '🔄 Running' if debugging_status == 'running' else '⏳ Pending'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔍 View Debug Information", expanded=(debugging_status == "running")):
            st.markdown("### 🐛 Debug Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Debug Iterations", state.iteration)
            with col2:
                error_count = len(state.errors) if state.errors else 0
                st.metric("Active Errors", error_count)
            with col3:
                st.metric("Current Status", state.current_node.value.title())
            
            # Show current errors being debugged
            if state.errors:
                st.markdown("### ⚠️ Current Issues Being Fixed")
                for i, error in enumerate(state.errors, 1):
                    st.error(f"**Error {i}:** {error}")
            
            # Show debugging logs with more detail
            debug_logs = [log for log in state.logs if log.node == "DEBUGGING"]
            if debug_logs:
                st.markdown("### 📋 Debug Process Log")
                
                for log in debug_logs[-10:]:  # Show last 10 debug logs
                    if log.level == "ERROR":
                        st.error(f"🔴 **{log.timestamp}** - {log.message}")
                    elif log.level == "SUCCESS":
                        st.success(f"🟢 **{log.timestamp}** - {log.message}")
                    elif log.level == "INFO":
                        st.info(f"🔵 **{log.timestamp}** - {log.message}")
                    elif log.level == "WARNING":
                        st.warning(f"🟡 **{log.timestamp}** - {log.message}")
                    else:
                        st.write(f"⚪ **{log.timestamp}** - {log.message}")
            
            # Show test results that caused debugging
            if state.test_results and ("FAILED" in state.test_results or "ERROR" in state.test_results):
                st.markdown("### 🧪 Failed Test Results")
                
                # Parse and highlight the failures
                test_lines = state.test_results.split('\n')
                failed_tests = []
                error_details = []
                
                for i, line in enumerate(test_lines):
                    if "FAILED" in line or "ERROR" in line:
                        failed_tests.append(line.strip())
                        # Try to get context
                        if i + 1 < len(test_lines) and test_lines[i + 1].strip():
                            error_details.append(test_lines[i + 1].strip())
                
                if failed_tests:
                    st.markdown("**Failed Tests:**")
                    for test in failed_tests[:5]:  # Show top 5 failures
                        st.code(test, language="text")
                
                if error_details:
                    st.markdown("**Error Details:**")
                    for detail in error_details[:3]:  # Show top 3 error details
                        st.code(detail, language="text")
                
                # Show full test output in expander
                with st.expander("📄 Full Test Output"):
                    st.code(state.test_results, language="text")

def display_logs(state: AgentState, unique_suffix=""):
    """Display real-time logs"""
    st.markdown("---")
    st.header("📋 Real-time Agent Logs")
    
    # Log controls with unique keys
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        show_details = st.checkbox("Show detailed logs", value=False, key=f"show_details_logs_{unique_suffix}")
    with col2:
        log_count = st.selectbox("Show last", [10, 20, 50, 100], index=1, key=f"log_count_select_{unique_suffix}")
    with col3:
        auto_scroll = st.checkbox("Auto-scroll", value=True, key=f"auto_scroll_logs_{unique_suffix}")
    
    if state.logs:
        # Create log display
        log_html = '<div class="log-container">'
        
        recent_logs = state.logs[-log_count:] if log_count else state.logs
        
        for log in recent_logs:
            level_emoji = {
                "INFO": "ℹ️",
                "DEBUG": "🔍", 
                "WARNING": "⚠️",
                "ERROR": "❌",
                "SUCCESS": "✅"
            }.get(log.level, "📝")
            
            level_colors = {
                "INFO": "#58a6ff",
                "DEBUG": "#8b949e", 
                "WARNING": "#d29922",
                "ERROR": "#f85149",
                "SUCCESS": "#3fb950"
            }
            
            color = level_colors.get(log.level, "#ffffff")
            
            log_html += f'''
            <div style="margin-bottom: 8px; color: {color};">
                <strong>{log.timestamp}</strong> {level_emoji} <strong>[{log.node}]</strong> {log.message}
            </div>
            '''
            
            if show_details and log.details:
                details_preview = log.details[:300] + "..." if len(log.details) > 300 else log.details
                log_html += f'''
                <div style="margin-left: 20px; margin-bottom: 5px; color: #8b949e; font-style: italic; font-size: 12px;">
                    {details_preview}
                </div>
                '''
        
        log_html += '</div>'
        
        # Add auto-scroll JavaScript
        if auto_scroll:
            log_html += '''
            <script>
            setTimeout(function() {
                var logContainer = document.querySelector('.log-container');
                if (logContainer) {
                    logContainer.scrollTop = logContainer.scrollHeight;
                }
            }, 100);
            </script>
            '''
        
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.info("🎯 Logs will appear here when the agent starts running...")

def main():
    st.set_page_config(
        page_title="Code Generation & Self-Repair Agent",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Code Generation & Self-Repair Agent")
    st.markdown("*AI-powered development tool that plans, codes, tests, and debugs automatically*")
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .phase-container {
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #0e1117;
    }
    .phase-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        font-weight: bold;
        font-size: 18px;
        color: white;
    }
    .phase-content {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 10px;
    }
    .status-pending { background-color: #ffd60a; color: #000; }
    .status-running { background-color: #0ea5e9; color: #fff; }
    .status-success { background-color: #22c55e; color: #fff; }
    .status-error { background-color: #ef4444; color: #fff; }
    .log-container {
        height: 300px;
        overflow-y: auto;
        background-color: #0a0a0a;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #262730;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.4;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .step-indicator {
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 10px;
        border-radius: 8px;
        background-color: #1a1a1a;
    }
    .step-number {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 15px;
    }
    .step-active { background-color: #0ea5e9; color: white; }
    .step-completed { background-color: #22c55e; color: white; }
    .step-pending { background-color: #6b7280; color: white; }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Get your API key from openrouter.ai"
        )
        
        model = st.selectbox(
            "Model",
            [
                "anthropic/claude-3-sonnet",
                "openai/gpt-4",
                "google/gemini-pro",
                "meta-llama/llama-2-70b-chat"
            ]
        )
        
        max_iterations = st.slider(
            "Max Debug Iterations",
            min_value=1,
            max_value=10,
            value=5
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Workflow Steps")
        
        # Workflow visualization
        if 'agent_state' in st.session_state and st.session_state.agent_state:
            state = st.session_state.agent_state
            steps = [
                ("Planning", NodeState.PLANNING),
                ("Coding", NodeState.CODING),
                ("Testing", NodeState.TESTING),
                ("Debugging", NodeState.DEBUGGING),
                ("Complete", NodeState.COMPLETE)
            ]
            
            for i, (step_name, step_state) in enumerate(steps):
                if step_state == state.current_node:
                    # Show current file being generated if in coding phase
                    if step_state == NodeState.CODING and hasattr(state, '_coding_file_index'):
                        current_file_idx = state._coding_file_index
                        if current_file_idx < len(state.planned_files):
                            current_file = state.planned_files[current_file_idx]
                            st.markdown(f"🔄 **{step_name}** (Current)")
                            st.markdown(f"   📝 Generating: `{current_file}`")
                        else:
                            st.markdown(f"🔄 **{step_name}** (Current)")
                    else:
                        st.markdown(f"🔄 **{step_name}** (Current)")
                elif get_phase_status(state, step_state) == "success":
                    st.markdown(f"✅ **{step_name}** (Done)")
                else:
                    st.markdown(f"⏳ {step_name} (Pending)")
                    
            # Show planned files if available
            if state.planned_files:
                st.markdown("---")
                st.markdown("### 📁 Planned Files")
                for filename in state.planned_files:
                    is_generated = any(cf.filename == filename for cf in state.code_files)
                    status = "✅" if is_generated else "⏳"
                    st.markdown(f"{status} `{filename}`")
        else:
            st.info("Workflow will appear here when agent starts")
    
    if not api_key:
        st.warning("Please enter your OpenRouter API key in the sidebar to get started.")
        st.stop()
    
    # Feature request input
    st.header("📝 Feature Request")
    feature_request = st.text_area(
        "Describe what you want to build:",
        height=120,
        placeholder="Example: Create a todo list app with the ability to add, delete, and mark tasks as complete. Include data persistence and a clean UI."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        generate_button = st.button("🚀 Generate Code", type="primary", disabled=not feature_request)
    
    with col2:
        if 'agent_state' in st.session_state and st.session_state.agent_state:
            if st.button("🔄 Reset Agent"):
                del st.session_state.agent_state
                st.rerun()
    
    # Agent execution
    if generate_button:
        # Initialize session state
        st.session_state.agent_state = AgentState(feature_request=feature_request, max_iterations=max_iterations)
        
        with st.spinner("Initializing AI agent..."):
            client = OpenRouterClient(api_key, model)
            agent = CodeAgent(client)
            
            try:
                # Execute agent with real-time updates
                state = st.session_state.agent_state
                
                node_functions = {
                    NodeState.PLANNING: agent.planning_node,
                    NodeState.CODING: agent.coding_node,
                    NodeState.TESTING: agent.testing_node,
                    NodeState.DEBUGGING: agent.debugging_node
                }
                
                # Create placeholder for dynamic updates
                status_placeholder = st.empty()
                
                step_counter = 0
                max_total_steps = 20  # Prevent infinite loops
                
                while state.current_node not in [NodeState.COMPLETE, NodeState.FAILED] and step_counter < max_total_steps:
                    if state.current_node in node_functions:
                        step_counter += 1
                        unique_id = f"step_{step_counter}_{int(time.time())}"
                        
                        # Update UI before executing node
                        with status_placeholder.container():
                            update_phase_ui(state, unique_id)
                            display_logs(state, unique_id)
                        
                        # Execute the current node
                        try:
                            state = node_functions[state.current_node](state)
                            st.session_state.agent_state = state
                            
                            # Update UI after executing node
                            unique_id_after = f"step_{step_counter}_after_{int(time.time())}"
                            with status_placeholder.container():
                                update_phase_ui(state, unique_id_after)
                                display_logs(state, unique_id_after)
                            
                        except Exception as e:
                            state.add_log("SYSTEM", "ERROR", f"Node execution failed: {str(e)}")
                            state.current_node = NodeState.FAILED
                            break
                        
                        # Small delay for UI updates
                        time.sleep(0.5)
                    else:
                        state.add_log("SYSTEM", "ERROR", f"Unknown node state: {state.current_node}")
                        break
                
                # Check if we hit max steps
                if step_counter >= max_total_steps:
                    state.add_log("SYSTEM", "ERROR", "Maximum execution steps reached")
                    state.current_node = NodeState.FAILED
                
                # Final update
                final_id = f"final_{int(time.time())}"
                with status_placeholder.container():
                    update_phase_ui(state, final_id)
                    display_logs(state, final_id)
                
                if state.current_node == NodeState.COMPLETE:
                    st.balloons()
                    st.success("🎉 Code generation completed successfully!")
                    
                    # Show final summary
                    if state.code_files:
                        st.info(f"✨ Generated {len(state.code_files)} files: {', '.join(cf.filename for cf in state.code_files)}")
                else:
                    st.error("❌ Maximum iterations reached. Code may need manual review.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display results if we have an agent state
    elif 'agent_state' in st.session_state and st.session_state.agent_state:
        state = st.session_state.agent_state
        
        # Check if we're still in the middle of coding
        if (state.current_node == NodeState.CODING and 
            hasattr(state, '_coding_file_index') and 
            state._coding_file_index < len(state.planned_files)):
            
            # Continue coding process automatically
            with st.spinner(f"Generating {state.planned_files[state._coding_file_index]}..."):
                try:
                    client = OpenRouterClient(api_key, model)
                    agent = CodeAgent(client)
                    
                    # Execute coding node
                    state = agent.coding_node(state)
                    st.session_state.agent_state = state
                    
                    # Refresh the page to show new file
                    st.rerun()
                    
                except Exception as e:
                    state.add_log("SYSTEM", "ERROR", f"Coding continuation failed: {str(e)}")
                    st.error(f"Error: {str(e)}")
        
        # Display current state
        static_id = f"static_{int(time.time())}"
        update_phase_ui(state, static_id)
        display_logs(state, static_id)

if __name__ == "__main__":
    main()