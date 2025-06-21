# Claude-Ollama MCP Server

A powerful Model Context Protocol (MCP) server that enables Claude Code instances to delegate tasks to local Ollama models. This creates a secure, private AI collaboration environment with comprehensive codebase analysis capabilities.

## Key Features

### **Task Delegation**
- Delegate coding tasks to local Ollama models for sensitive work
- 8 specialized delegation tools for different coding scenarios
- Maintains privacy by keeping code local to your machine

### **Autonomous Codebase Analysis** 
- **NEW**: Ollama models can independently explore and analyze entire codebases
- File system access with security boundaries
- Interactive analysis sessions with tool usage
- Comprehensive project understanding and reporting

### **Advanced Logging & Monitoring**
- Session-specific conversation logs with timestamps
- Rotating log files with configurable size limits
- Debug logging for troubleshooting
- Real-time console output during analysis

### **Secure File Operations**
- Path validation and sandboxing
- Safe bash command execution with allowlists
- File size and result limits for performance
- Read-only operations with controlled access

## Architecture

The system consists of three main components:

1. **`claude_ollama_mcp.py`** - Main MCP server with 9 delegation tools and file system access
2. **`claude_ollama_bridge.py`** - Simple direct bridge for basic Ollama communication  
3. **`ollama_delegate.py`** - Standalone command-line delegation utility

### Core Classes

- **`OllamaDelegator`** - Handles task delegation with specialized system prompts
- **`FileSystemTools`** - Secure file operations for Ollama models
- **MCP Server** - Provides tools via Model Context Protocol

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Claude Code MCP Server

Add this to your Claude Code MCP configuration:

**macOS/Linux**: `~/.config/claude-code/mcp_servers.json`  
**Windows**: `%APPDATA%\claude-code\mcp_servers.json`

```json
{
  "mcpServers": {
    "claude-ollama": {
      "command": "python",
      "args": ["/absolute/path/to/claude_ollama_mcp.py"],
      "env": {}
    }
  }
}
```

### 3. Start Ollama
```bash
# Make sure Ollama is running with a coding model
ollama serve
ollama pull qwen2.5-coder:latest
```

### 4. Restart Claude Code
The MCP server will automatically connect when Claude Code loads.

## Available Tools

### **Codebase Analysis**
- **`delegate_analyze_codebase`** - **NEW**: Comprehensive project analysis with file system access
  - Allows Ollama to independently explore directories and files
  - Generates detailed project reports and architecture analysis
  - Interactive analysis sessions with multiple iterations

### **Code Tasks**
- **`delegate_code_review`** - Review code for bugs, security issues, and improvements
- **`delegate_generate_tests`** - Generate comprehensive unit tests with good coverage
- **`delegate_refactor`** - Restructure code while maintaining functionality
- **`delegate_implement`** - Implement new features from specifications
- **`delegate_document`** - Add documentation, comments, and docstrings
- **`delegate_explain`** - Explain complex code logic and algorithms
- **`delegate_debug`** - Analyze and fix problematic code
- **`delegate_optimize`** - Improve code performance and efficiency

## Usage Examples

### Basic Task Delegation
```
Claude: I need to review this authentication code for security issues.

[Claude automatically uses delegate_code_review with your code]
[Local Ollama model analyzes the code privately]
[Results returned with security recommendations]
```

### Autonomous Codebase Analysis
```
Claude: Please analyze this entire project and create a comprehensive overview.

[Claude uses delegate_analyze_codebase]
[Ollama explores directories, reads key files, analyzes structure]
[Generates detailed project report with architecture insights]
```

### Command Line Usage
```bash
# Test basic delegation
python claude_ollama_bridge.py qwen2.5-coder:latest "Explain this Python function" "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"

# Test specific delegation types
python ollama_delegate.py code_review "Review this function for bugs" "function code here"
python ollama_delegate.py generate_tests "Create tests for this class" "class code here"
```

## Logging & Monitoring

The server creates detailed logs in the `logs/` directory:

- **`claude_ollama_mcp.log`** - Main application log (rotating, 10MB limit)
- **`debug.log`** - Detailed debug information (rotating, 50MB limit)  
- **`conversation_TIMESTAMP.log`** - Session-specific conversation logs
- **Console output** - Real-time analysis progress (warnings/errors only)

### Log Structure
```
2024-06-21 10:30:45 - claude-ollama-mcp - INFO - analyze_codebase_with_tools:234 - Starting analysis
2024-06-21 10:30:46 - INFO - OLLAMA_RESPONSE_LENGTH: 1247 chars
2024-06-21 10:30:47 - INFO - TOOL_EXECUTE: fs_list
2024-06-21 10:30:47 - INFO - TOOL_RESULT_LENGTH: 892 chars
```

## Security Features

### File System Security
- **Path validation** - All paths confined to project directory
- **Safe commands** - Only allowlisted bash commands permitted
- **Size limits** - File size and result count restrictions
- **Read-only access** - No file modifications allowed

### Privacy Protection
- **Local processing** - All analysis happens on your machine
- **No data transmission** - Code never leaves your environment
- **Session isolation** - Each analysis session is independent

## Testing & Development

### Test MCP Server
```bash
python claude_ollama_mcp.py --help
```

### Test Direct Bridge
```bash
python claude_ollama_bridge.py qwen2.5-coder:latest "Hello, world!" "You are a helpful assistant"
```

### Test Delegation
```bash
python ollama_delegate.py explain "What does this code do?" "print('Hello, world!')"
```

## Requirements

- **Ollama** running on `localhost:11434`
- **Python 3.8+** with required packages
- **Coding model** like `qwen2.5-coder:latest`, `codellama:latest`, or `deepseek-coder:latest`
- **Claude Code** with MCP support

### Python Dependencies
```
mcp>=1.0.0
requests>=2.25.0
```

## Configuration

### Environment Variables
- `OLLAMA_BASE_URL` - Override default Ollama URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL` - Override default model (default: `qwen2.5-coder:latest`)

### File System Tools Configuration
- `max_file_size` - Maximum file size to read (default: 1MB)
- `max_results` - Maximum search results (default: 100)
- `timeout` - Command timeout in seconds (default: 10)

## Troubleshooting

### Common Issues

1. **MCP Server Not Loading**
   - Check file paths in configuration are absolute
   - Verify Python environment has required packages
   - Check Claude Code logs for error messages

2. **Ollama Connection Failed**
   - Confirm Ollama is running: `ollama serve`
   - Test API: `curl http://localhost:11434/api/tags`
   - Check firewall settings

3. **Analysis Not Working**
   - Check `logs/debug.log` for detailed error information
   - Verify model supports tool usage
   - Ensure sufficient system resources

### Debug Mode
Set logging level to DEBUG in the code for verbose output:
```python
root_logger.setLevel(logging.DEBUG)
```

## Contributing

This is a powerful foundation for AI-assisted development. The codebase analysis feature opens up possibilities for:

- Automated code reviews
- Documentation generation
- Architecture analysis
- Security auditing
- Code quality metrics
- Migration planning

## License

This project enables secure, private AI collaboration for sensitive codebases while maintaining full control over your data and development environment.
