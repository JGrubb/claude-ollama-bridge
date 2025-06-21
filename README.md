# Claude-Ollama MCP Server

This MCP server allows Claude instances to delegate coding tasks to your local Ollama models, enabling secure, private AI collaboration.

## Setup

1. **Install dependencies** (already done):
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Claude Code to use the MCP server**:
   
   Add this to your Claude Code MCP configuration file:
   
   **For macOS/Linux**: `~/.config/claude-code/mcp_servers.json`
   **For Windows**: `%APPDATA%\claude-code\mcp_servers.json`
   
   ```json
   {
     "mcpServers": {
       "claude-ollama": {
         "command": "python",
         "args": ["/Users/johngrubb/play/Ollama-stuff/claude_ollama_mcp.py"],
         "env": {}
       }
     }
   }
   ```

3. **Restart Claude Code** to load the new MCP server

## Available Tools

The MCP server provides these delegation tools:

- `delegate_code_review` - Review code for bugs and improvements
- `delegate_generate_tests` - Generate comprehensive unit tests
- `delegate_refactor` - Refactor code with specific instructions
- `delegate_implement` - Implement new code from specifications
- `delegate_document` - Add documentation and comments
- `delegate_explain` - Explain how code works
- `delegate_debug` - Debug problematic code
- `delegate_optimize` - Optimize code for performance

## Usage

Once configured, any Claude Code instance can use these tools from any directory:

```
I need to review this sensitive code. Let me delegate this to your local model...
[Claude automatically uses delegate_code_review tool]
```

## Requirements

- Ollama running on localhost:11434
- At least one coding model (like qwen2.5-coder:latest)
- Python 3.8+
- mcp and requests packages

## Testing

To test if the server works:
```bash
python claude_ollama_mcp.py --help
```

The server should start and connect to Ollama automatically when Claude Code loads it.