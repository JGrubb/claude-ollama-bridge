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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude-ollama-mcp")

class OllamaDelegator:
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen2.5-coder:latest"):
        self.base_url = base_url
        self.default_model = default_model
    
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
            "optimize": "You are a performance optimization expert. Improve code efficiency and performance."
        }
        
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
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "delegate_code_review":
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