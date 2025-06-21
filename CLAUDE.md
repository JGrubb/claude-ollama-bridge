# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

This is a Claude-Ollama MCP (Model Context Protocol) server that enables Claude Code instances to delegate tasks to local Ollama models. The system consists of:

- **claude_ollama_mcp.py**: Main MCP server implementation with 8 delegation tools
- **claude_ollama_bridge.py**: Simple direct bridge for basic Ollama communication
- **ollama_delegate.py**: Standalone delegation utility for command-line usage
- **mcp_config.json**: MCP server configuration for Claude Code

### Core Components

1. **OllamaDelegator class** (claude_ollama_mcp.py:28): Handles task delegation with specialized system prompts for different task types
2. **MCP Server** (claude_ollama_mcp.py:81): Provides 8 delegation tools via MCP protocol
3. **OllamaBridge class** (claude_ollama_bridge.py:12): Basic HTTP client for Ollama API

## Development Commands

### Setup
```bash
pip install -r requirements.txt
```

### Testing MCP Server
```bash
python claude_ollama_mcp.py --help
```

### Testing Basic Bridge
```bash
python claude_ollama_bridge.py <model> <prompt> [system_prompt]
```

### Testing Delegation
```bash
python ollama_delegate.py <task_type> <prompt> [context]
# Task types: code_review, generate_tests, refactor, document, explain, implement
```

## MCP Integration

The server expects Ollama running on `localhost:11434` with coding models like `qwen2.5-coder:latest`. Configure Claude Code with:

```json
{
  "mcpServers": {
    "claude-ollama": {
      "command": "python",
      "args": ["/path/to/claude_ollama_mcp.py"],
      "env": {}
    }
  }
}
```

## Available Delegation Tools

- `delegate_code_review`: Code analysis and bug detection
- `delegate_generate_tests`: Unit test generation
- `delegate_refactor`: Code restructuring
- `delegate_implement`: New code implementation
- `delegate_document`: Documentation generation
- `delegate_explain`: Code explanation
- `delegate_debug`: Error analysis and fixes
- `delegate_optimize`: Performance optimization

Each tool accepts code, language (default: python), and optional context parameters.