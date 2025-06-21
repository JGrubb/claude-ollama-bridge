#!/usr/bin/env python3
"""
Claude-Ollama MCP Server
Allows Claude instances to delegate tasks to local Ollama models
"""

import asyncio
import json
import logging
from typing import Any, Sequence, Dict, List, Optional
import os
import subprocess
import glob
import re
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

import requests
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    Tool,
    TextContent,
)

# Setup structured logging with file outputs
def setup_logging():
    """Setup comprehensive logging to files and console"""
    # Ensure logs directory exists
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Formatter for detailed logs
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Formatter for conversation logs
    conversation_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 1. Main application log (rotating)
    main_handler = RotatingFileHandler(
        log_dir / "claude_ollama_mcp.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(detailed_formatter)
    
    # 2. Debug log (rotating)
    debug_handler = RotatingFileHandler(
        log_dir / "debug.log",
        maxBytes=50*1024*1024,  # 50MB
        backupCount=3
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)
    
    # 3. Session-specific conversation log
    conversation_handler = logging.FileHandler(
        log_dir / f"conversation_{timestamp}.log"
    )
    conversation_handler.setLevel(logging.INFO)
    conversation_handler.setFormatter(conversation_formatter)
    
    # 4. Console handler (for immediate feedback)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    # Add handlers to root logger
    root_logger.addHandler(main_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(conversation_handler)
    root_logger.addHandler(console_handler)
    
    # Create specific loggers
    app_logger = logging.getLogger("claude-ollama-mcp")
    conversation_logger = logging.getLogger("claude-ollama-mcp.conversation")
    
    # Suppress noisy third-party logs in files
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    return app_logger, conversation_logger, timestamp

# Initialize logging
logger, conversation_logger, session_id = setup_logging()

class FileSystemTools:
    """Secure file system operations for Ollama models"""
    
    def __init__(self, base_path: str = ".", max_file_size: int = 1000000, max_results: int = 100):
        self.base_path = Path(base_path).resolve()
        self.max_file_size = max_file_size
        self.max_results = max_results
        
        # Safe bash commands allowlist
        self.safe_commands = {
            'ls', 'find', 'grep', 'head', 'tail', 'wc', 'cat', 'file', 'which',
            'pwd', 'echo', 'date', 'whoami', 'id', 'uname', 'df', 'du', 'ps'
        }
    
    def _validate_path(self, path: str) -> Path:
        """Validate and resolve path within base directory"""
        try:
            full_path = (self.base_path / path).resolve()
            # Ensure path is within base directory
            full_path.relative_to(self.base_path)
            return full_path
        except (ValueError, OSError):
            raise ValueError(f"Invalid or unsafe path: {path}")
    
    def read_file(self, path: str, max_lines: Optional[int] = None) -> str:
        """Read file contents with safety checks"""
        file_path = self._validate_path(path)
        
        if not file_path.exists():
            return f"File not found: {path}"
        
        if not file_path.is_file():
            return f"Path is not a file: {path}"
        
        if file_path.stat().st_size > self.max_file_size:
            return f"File too large (>{self.max_file_size} bytes): {path}"
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            lines.append(f"... (truncated at {max_lines} lines)")
                            break
                        lines.append(line.rstrip())
                    return '\n'.join(lines)
                else:
                    return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def list_directory(self, path: str = ".", show_hidden: bool = False) -> str:
        """List directory contents"""
        dir_path = self._validate_path(path)
        
        if not dir_path.exists():
            return f"Directory not found: {path}"
        
        if not dir_path.is_dir():
            return f"Path is not a directory: {path}"
        
        try:
            items = []
            for item in sorted(dir_path.iterdir()):
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                item_type = "DIR" if item.is_dir() else "FILE"
                relative_path = item.relative_to(self.base_path)
                size = item.stat().st_size if item.is_file() else 0
                items.append(f"{item_type:4} {size:>8} {relative_path}")
                
                if len(items) >= self.max_results:
                    items.append(f"... (truncated at {self.max_results} items)")
                    break
            
            return '\n'.join(items) if items else "Directory is empty"
            
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def glob_files(self, pattern: str, recursive: bool = True) -> str:
        """Find files matching pattern"""
        try:
            # Validate pattern doesn't escape base directory
            if '..' in pattern or pattern.startswith('/'):
                return "Invalid pattern: cannot escape base directory"
            
            search_pattern = str(self.base_path / pattern)
            
            if recursive:
                matches = glob.glob(search_pattern, recursive=True)
            else:
                matches = glob.glob(search_pattern)
            
            # Convert to relative paths and limit results
            relative_matches = []
            for match in sorted(matches):
                try:
                    rel_path = Path(match).relative_to(self.base_path)
                    relative_matches.append(str(rel_path))
                    
                    if len(relative_matches) >= self.max_results:
                        relative_matches.append(f"... (truncated at {self.max_results} results)")
                        break
                except ValueError:
                    continue
            
            return '\n'.join(relative_matches) if relative_matches else "No matches found"
            
        except Exception as e:
            return f"Error in glob search: {str(e)}"
    
    def grep_files(self, pattern: str, file_pattern: str = "*", max_matches: int = 50) -> str:
        """Search for pattern in files"""
        try:
            if '..' in file_pattern or file_pattern.startswith('/'):
                return "Invalid file pattern: cannot escape base directory"
            
            search_path = str(self.base_path / file_pattern)
            files = glob.glob(search_path, recursive=True)
            
            matches = []
            for file_path in files:
                if len(matches) >= max_matches:
                    matches.append(f"... (truncated at {max_matches} matches)")
                    break
                
                try:
                    path_obj = Path(file_path)
                    if not path_obj.is_file():
                        continue
                    
                    rel_path = path_obj.relative_to(self.base_path)
                    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if re.search(pattern, line, re.IGNORECASE):
                                matches.append(f"{rel_path}:{line_num}: {line.strip()}")
                                
                                if len(matches) >= max_matches:
                                    break
                                    
                except Exception:
                    continue
            
            return '\n'.join(matches) if matches else "No matches found"
            
        except Exception as e:
            return f"Error in grep search: {str(e)}"
    
    def execute_command(self, command: str, timeout: int = 10) -> str:
        """Execute safe bash commands"""
        try:
            # Parse command and validate
            cmd_parts = command.strip().split()
            if not cmd_parts:
                return "Empty command"
            
            base_cmd = cmd_parts[0]
            if base_cmd not in self.safe_commands:
                return f"Command not allowed: {base_cmd}. Allowed: {', '.join(sorted(self.safe_commands))}"
            
            # Execute in base directory
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            return output[:5000]  # Limit output size
            
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"

class OllamaDelegator:
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen2.5-coder:latest", verbose_stdout: bool = True):
        self.base_url = base_url
        self.default_model = default_model
        self.fs_tools = FileSystemTools()
        self.verbose_stdout = verbose_stdout
        
        # File system tools definitions for Ollama
        self.ollama_tools = [
            {
                "type": "function",
                "function": {
                    "name": "fs_read",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to project root"},
                            "max_lines": {"type": "integer", "description": "Maximum lines to read (optional)"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "fs_list",
                    "description": "List contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path (default: current directory)"},
                            "show_hidden": {"type": "boolean", "description": "Show hidden files"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fs_glob", 
                    "description": "Find files matching a pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py')"},
                            "recursive": {"type": "boolean", "description": "Search recursively"}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fs_grep",
                    "description": "Search for text pattern in files", 
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Text pattern to search for"},
                            "file_pattern": {"type": "string", "description": "File pattern to search in (default: all files)"},
                            "max_matches": {"type": "integer", "description": "Maximum matches to return"}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fs_bash",
                    "description": "Execute safe bash commands",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "command": {"type": "string", "description": "Bash command to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds"}
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
    
    def execute_fs_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute file system tool and return result"""
        logger.debug(f"Executing file system tool: {tool_name} with arguments: {arguments}")
        try:
            if tool_name == "fs_read":
                return self.fs_tools.read_file(
                    arguments["path"], 
                    arguments.get("max_lines")
                )
            elif tool_name == "fs_list":
                return self.fs_tools.list_directory(
                    arguments.get("path", "."),
                    arguments.get("show_hidden", False)
                )
            elif tool_name == "fs_glob":
                return self.fs_tools.glob_files(
                    arguments["pattern"],
                    arguments.get("recursive", True)
                )
            elif tool_name == "fs_grep":
                return self.fs_tools.grep_files(
                    arguments["pattern"],
                    arguments.get("file_pattern", "*"),
                    arguments.get("max_matches", 50)
                )
            elif tool_name == "fs_bash":
                return self.fs_tools.execute_command(
                    arguments["command"],
                    arguments.get("timeout", 10)
                )
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool execution error: {str(e)}"
    
    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from Ollama response content"""
        tool_calls = []
        
        try:
            # Look for JSON blocks in the content
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    tool_call = json.loads(match)
                    if "name" in tool_call:
                        tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue
            
            # Also try to parse direct JSON (fallback)
            if not tool_calls:
                try:
                    if content.strip().startswith('{'):
                        tool_call = json.loads(content.strip())
                        if "name" in tool_call:
                            tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            logger.warning(f"Error extracting tool calls: {e}")
        
        return tool_calls
    
    def analyze_codebase_with_tools(self, request: str, max_iterations: int = 10) -> str:
        """Allow Ollama to analyze codebase using file system tools"""
        conversation = [
            {
                "role": "system",
                "content": """You are a codebase analysis expert. You have access to file system tools to explore and analyze projects.

Available tools:
- fs_read: Read file contents
- fs_list: List directory contents  
- fs_glob: Find files by pattern
- fs_grep: Search text in files
- fs_bash: Execute safe commands

Start by exploring the project structure, then read key files to understand the codebase. Provide comprehensive analysis. When you want to use a tool, respond with JSON like: {"name": "fs_list", "arguments": {"path": "."}}."""
            },
            {
                "role": "user", 
                "content": request
            }
        ]
        
        logger.info(f"=== STARTING CODEBASE ANALYSIS ===")
        logger.info(f"Request: {request}")
        logger.info(f"Initial conversation: {json.dumps(conversation, indent=2)}")
        
        # Log session start
        conversation_logger.info(f"SESSION_START: {session_id}")
        conversation_logger.info(f"REQUEST: {request}")
        conversation_logger.info(f"MODEL: {self.default_model}")
        conversation_logger.info(f"MAX_ITERATIONS: {max_iterations}")
        
        if self.verbose_stdout:
            print("\n" + "="*60)
            print("🤖 CLAUDE-OLLAMA CODEBASE ANALYSIS SESSION")
            print("="*60)
            print(f"📝 Request: {request}")
            print(f"🎯 Model: {self.default_model}")
            print(f"🔄 Max iterations: {max_iterations}")
            print(f"📁 Session ID: {session_id}")
        
        for iteration in range(max_iterations):
            logger.info(f"Analysis iteration {iteration + 1}")
            
            # Send request to Ollama with tools
            try:
                request_payload = {
                    "model": self.default_model,
                    "messages": conversation,
                    "tools": self.ollama_tools,
                    "stream": False
                }
                
                logger.info(f"\n=== ITERATION {iteration + 1} REQUEST TO OLLAMA ===")
                logger.info(f"Request payload: {json.dumps(request_payload, indent=2)}")
                
                # Log iteration start
                conversation_logger.info(f"ITERATION_{iteration + 1}_START")
                conversation_logger.info(f"SENDING_TO_OLLAMA: {len(conversation)} messages")
                
                if self.verbose_stdout:
                    print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
                    print("📤 Sending request to Ollama...")
                
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=request_payload,
                    timeout=60
                )
                
                logger.info(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    return f"Ollama API error: {response.status_code}"
                
                result = response.json()
                logger.info(f"\n=== OLLAMA RESPONSE ===")
                logger.info(f"Full response: {json.dumps(result, indent=2)}")
                
                assistant_message = result["message"]
                conversation.append(assistant_message)
                
                # Check if model wants to use tools
                content = assistant_message.get("content", "")
                logger.info(f"\n=== ASSISTANT MESSAGE CONTENT ===")
                logger.info(f"Content: {content}")
                
                # Log Ollama response
                conversation_logger.info(f"OLLAMA_RESPONSE_LENGTH: {len(content)} chars")
                conversation_logger.info(f"OLLAMA_RESPONSE: {content[:500]}{'...' if len(content) > 500 else ''}")
                
                if self.verbose_stdout:
                    print(f"\n🤖 Ollama Response:")
                    print("-" * 40)
                    print(content)
                    print("-" * 40)
                
                # Parse JSON tool calls from content
                tool_calls = self._extract_tool_calls(content)
                logger.info(f"\n=== EXTRACTED TOOL CALLS ===")
                logger.info(f"Tool calls: {json.dumps(tool_calls, indent=2)}")
                
                if not tool_calls:
                    # No more tools needed, return final response
                    conversation_logger.info(f"ANALYSIS_COMPLETE_ITERATION_{iteration + 1}")
                    conversation_logger.info(f"FINAL_RESULT_LENGTH: {len(content)}")
                    conversation_logger.info(f"FINAL_RESULT: {content}")
                    conversation_logger.info(f"SESSION_END: {session_id}")
                    
                    if self.verbose_stdout:
                        print(f"\n🎯 Final Analysis Complete!")
                        print("="*60)
                        print("📊 FINAL RESULT:")
                        print("="*60)
                        print(content)
                        print("="*60)
                    return content
                
                # Execute requested tools
                for i, tool_call in enumerate(tool_calls):
                    tool_name = tool_call.get("name")
                    arguments = tool_call.get("arguments", {})
                    
                    logger.info(f"\n=== EXECUTING TOOL {i+1}/{len(tool_calls)} ===")
                    logger.info(f"Tool: {tool_name}")
                    logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")
                    
                    # Log tool execution
                    conversation_logger.info(f"TOOL_EXECUTE: {tool_name}")
                    conversation_logger.info(f"TOOL_ARGS: {json.dumps(arguments)}")
                    
                    if self.verbose_stdout:
                        print(f"\n🔧 Executing Tool {i+1}/{len(tool_calls)}: {tool_name}")
                        print(f"   Arguments: {json.dumps(arguments)}")
                    
                    tool_result = self.execute_fs_tool(tool_name, arguments)
                    
                    logger.info(f"\n=== TOOL RESULT ===")
                    logger.info(f"Result length: {len(tool_result)} characters")
                    logger.info(f"Result preview: {tool_result[:500]}..." if len(tool_result) > 500 else f"Result: {tool_result}")
                    
                    # Log tool result
                    conversation_logger.info(f"TOOL_RESULT_LENGTH: {len(tool_result)}")
                    conversation_logger.info(f"TOOL_RESULT: {tool_result[:200]}{'...' if len(tool_result) > 200 else ''}")
                    
                    if self.verbose_stdout:
                        print(f"✅ Tool Result ({len(tool_result)} chars):")
                        if len(tool_result) > 300:
                            print(f"   {tool_result[:300]}...")
                        else:
                            print(f"   {tool_result}")
                    
                    # Add tool result to conversation
                    tool_message = {
                        "role": "tool",
                        "content": tool_result
                    }
                    conversation.append(tool_message)
                    logger.info(f"Added tool message to conversation: {json.dumps(tool_message, indent=2)}")
                
            except Exception as e:
                logger.error(f"Error in analysis iteration {iteration + 1}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return f"Error in analysis iteration {iteration + 1}: {str(e)}"
        
        logger.info(f"\n=== ANALYSIS COMPLETED ===")
        logger.info(f"Reached max iterations ({max_iterations})")
        
        conversation_logger.info(f"MAX_ITERATIONS_REACHED: {max_iterations}")
        conversation_logger.info(f"SESSION_END: {session_id}")
        
        if self.verbose_stdout:
            print(f"\n⏰ Analysis completed (reached max iterations: {max_iterations})")
            print("="*60)
        
        return "Analysis completed (max iterations reached)"
    
    def delegate_task(self, task_type: str, prompt: str, context: str = "", model: str = None) -> str:
        """Delegate a task to Ollama model"""
        if model is None:
            model = self.default_model
            
        # Task-specific system prompts
        system_prompts = {
            "code_review": "You are an expert code reviewer. Analyze code for bugs, improvements, and best practices. Be concise but thorough.",
            "generate_tests": "You are a testing expert. Generate comprehensive unit tests using appropriate frameworks.",
            "refactor": "You are a refactoring expert. Improve code structure while maintaining functionality.",
            "document": "You are a documentation expert. Add clear, helpful comments and docstrings.",
            "explain": "You are a code explanation expert. Explain code clearly and concisely.",
            "implement": "You are a coding expert. Write clean, efficient, well-structured code.",
            "debug": "You are a debugging expert. Analyze errors and suggest fixes with clear explanations.",
            "optimize": "You are a performance optimization expert. Improve code efficiency and performance.",
            "analyze_codebase": "You are a codebase analysis expert with file system access. Explore and analyze the project comprehensively."
        }
        
        # Special handling for codebase analysis
        if task_type == "analyze_codebase":
            return self.analyze_codebase_with_tools(prompt)
        
        system_prompt = system_prompts.get(task_type, "You are a helpful coding assistant.")
        
        # Add context if provided
        if context:
            system_prompt += f"\n\nProject context: {context}"
        
        # Prepare request
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return f"Error communicating with Ollama: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Unexpected error: {str(e)}"

# Create global delegator instance
ollama_delegator = OllamaDelegator()

# Create MCP server
server = Server("claude-ollama-mcp")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for Claude-Ollama delegation"""
    return [
        # New codebase analysis tool
        Tool(
            name="delegate_analyze_codebase",
            description="Delegate comprehensive codebase analysis to local Ollama model with file system access. The model can independently explore directories, read files, and perform detailed analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "Analysis request (e.g., 'Create a comprehensive project overview', 'Analyze the architecture', 'Find potential issues')"
                    },
                    "base_path": {
                        "type": "string",
                        "description": "Base directory for analysis (optional, defaults to current directory)",
                        "default": "."
                    }
                },
                "required": ["request"]
            }
        ),
        Tool(
            name="delegate_code_review",
            description="Delegate code review to local Ollama model. Analyzes code for bugs, improvements, and best practices.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to review"
                    },
                    "language": {
                        "type": "string", 
                        "description": "Programming language (optional)",
                        "default": "python"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional project context (optional)",
                        "default": ""
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="delegate_generate_tests",
            description="Delegate test generation to local Ollama model. Creates comprehensive unit tests.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to generate tests for"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (optional)",
                        "default": "python"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional project context (optional)",
                        "default": ""
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="delegate_refactor",
            description="Delegate code refactoring to local Ollama model. Improves code structure while maintaining functionality.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to refactor"
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Specific refactoring instructions"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (optional)",
                        "default": "python"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional project context (optional)",
                        "default": ""
                    }
                },
                "required": ["code", "instructions"]
            }
        ),
        Tool(
            name="delegate_implement",
            description="Delegate code implementation to local Ollama model. Creates new code based on specifications.",
            inputSchema={
                "type": "object",
                "properties": {
                    "specification": {
                        "type": "string",
                        "description": "What to implement (function, class, feature, etc.)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (optional)",
                        "default": "python"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional project context (optional)",
                        "default": ""
                    }
                },
                "required": ["specification"]
            }
        ),
        Tool(
            name="delegate_document",
            description="Delegate documentation generation to local Ollama model. Adds comments and docstrings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to document"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (optional)",
                        "default": "python"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional project context (optional)",
                        "default": ""
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="delegate_explain",
            description="Delegate code explanation to local Ollama model. Provides clear explanations of how code works.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to explain"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (optional)",
                        "default": "python"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional project context (optional)",
                        "default": ""
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="delegate_debug",
            description="Delegate debugging to local Ollama model. Analyzes errors and suggests fixes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The problematic code"
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message or description of the problem"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (optional)",
                        "default": "python"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional project context (optional)",
                        "default": ""
                    }
                },
                "required": ["code", "error"]
            }
        ),
        Tool(
            name="delegate_optimize",
            description="Delegate performance optimization to local Ollama model. Improves code efficiency.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to optimize"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language (optional)",
                        "default": "python"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional project context (optional)",
                        "default": ""
                    }
                },
                "required": ["code"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    logger.info(f"\n=== MCP TOOL CALL ===")
    logger.info(f"Tool: {name}")
    logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")
    
    try:
        if name == "delegate_analyze_codebase":
            request = arguments["request"]
            base_path = arguments.get("base_path", ".")
            
            # Update file system tools base path if needed
            if base_path != ".":
                ollama_delegator.fs_tools = FileSystemTools(base_path)
            
            result = ollama_delegator.analyze_codebase_with_tools(request)
            logger.info(f"\n=== MCP TOOL RESULT ===")
            logger.info(f"Result length: {len(result)} characters")
            logger.info(f"Result preview: {result[:300]}..." if len(result) > 300 else f"Result: {result}")
            
        elif name == "delegate_code_review":
            code = arguments["code"]
            language = arguments.get("language", "python")
            context = arguments.get("context", "")
            
            prompt = f"Review this {language} code:\n\n```{language}\n{code}\n```"
            result = ollama_delegator.delegate_task("code_review", prompt, context)
            
        elif name == "delegate_generate_tests":
            code = arguments["code"]
            language = arguments.get("language", "python")
            context = arguments.get("context", "")
            
            prompt = f"Generate comprehensive unit tests for this {language} code:\n\n```{language}\n{code}\n```"
            result = ollama_delegator.delegate_task("generate_tests", prompt, context)
            
        elif name == "delegate_refactor":
            code = arguments["code"]
            instructions = arguments["instructions"]
            language = arguments.get("language", "python")
            context = arguments.get("context", "")
            
            prompt = f"Refactor this {language} code according to these instructions: {instructions}\n\n```{language}\n{code}\n```"
            result = ollama_delegator.delegate_task("refactor", prompt, context)
            
        elif name == "delegate_implement":
            specification = arguments["specification"]
            language = arguments.get("language", "python")
            context = arguments.get("context", "")
            
            prompt = f"Implement this in {language}: {specification}"
            result = ollama_delegator.delegate_task("implement", prompt, context)
            
        elif name == "delegate_document":
            code = arguments["code"]
            language = arguments.get("language", "python")
            context = arguments.get("context", "")
            
            prompt = f"Add comprehensive documentation to this {language} code:\n\n```{language}\n{code}\n```"
            result = ollama_delegator.delegate_task("document", prompt, context)
            
        elif name == "delegate_explain":
            code = arguments["code"]
            language = arguments.get("language", "python")
            context = arguments.get("context", "")
            
            prompt = f"Explain how this {language} code works:\n\n```{language}\n{code}\n```"
            result = ollama_delegator.delegate_task("explain", prompt, context)
            
        elif name == "delegate_debug":
            code = arguments["code"]
            error = arguments["error"]
            language = arguments.get("language", "python")
            context = arguments.get("context", "")
            
            prompt = f"Debug this {language} code that has the following error: {error}\n\n```{language}\n{code}\n```"
            result = ollama_delegator.delegate_task("debug", prompt, context)
            
        elif name == "delegate_optimize":
            code = arguments["code"]
            language = arguments.get("language", "python")
            context = arguments.get("context", "")
            
            prompt = f"Optimize this {language} code for better performance:\n\n```{language}\n{code}\n```"
            result = ollama_delegator.delegate_task("optimize", prompt, context)
            
        else:
            result = f"Unknown tool: {name}"
            
        return [TextContent(type="text", text=result)]
        
    except Exception as e:
        logger.error(f"Error in tool call: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Run the MCP server"""
    logger.info("Starting Claude-Ollama MCP Server")
    
    # Test Ollama connection
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"Connected to Ollama. Available models: {[m['name'] for m in models]}")
        else:
            logger.warning("Ollama API responded but with error status")
    except Exception as e:
        logger.error(f"Could not connect to Ollama: {e}")
        logger.error("Make sure Ollama is running on localhost:11434")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="claude-ollama-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())