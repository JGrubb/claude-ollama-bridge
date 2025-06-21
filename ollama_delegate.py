#!/usr/bin/env python3
"""
Simple Claude-Ollama Delegation Bridge
Proof of concept for Claude orchestrating local Ollama models
"""

import requests
import json
import sys

def delegate_to_ollama(task_type, prompt, context="", model="qwen2.5-coder:latest"):
    """
    Delegate a task to local Ollama model
    
    Args:
        task_type: Type of task (code_review, generate_tests, refactor, etc.)
        prompt: The specific task prompt
        context: Additional context about the project/code
        model: Ollama model to use
    """
    
    # Task-specific system prompts
    system_prompts = {
        "code_review": "You are an expert code reviewer. Analyze code for bugs, improvements, and best practices. Be concise but thorough.",
        "generate_tests": "You are a testing expert. Generate comprehensive unit tests using appropriate frameworks.",
        "refactor": "You are a refactoring expert. Improve code structure while maintaining functionality.",
        "document": "You are a documentation expert. Add clear, helpful comments and docstrings.",
        "explain": "You are a code explanation expert. Explain code clearly and concisely.",
        "implement": "You are a coding expert. Write clean, efficient, well-structured code."
    }
    
    system_prompt = system_prompts.get(task_type, "You are a helpful coding assistant.")
    
    # Add context if provided
    if context:
        system_prompt += f"\n\nProject context: {context}"
    
    # Prepare request
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    if len(sys.argv) < 3:
        print("Usage: python ollama_delegate.py <task_type> <prompt> [context]")
        print("Task types: code_review, generate_tests, refactor, document, explain, implement")
        sys.exit(1)
    
    task_type = sys.argv[1]
    prompt = sys.argv[2]
    context = sys.argv[3] if len(sys.argv) > 3 else ""
    
    result = delegate_to_ollama(task_type, prompt, context)
    print(result)

if __name__ == "__main__":
    main()