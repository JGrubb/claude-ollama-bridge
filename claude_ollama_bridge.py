#!/usr/bin/env python3
"""
Simple Claude-Ollama Bridge
Allows Claude to delegate tasks to local Ollama models
"""

import requests
import json
import sys
from typing import Optional, Dict, Any

class OllamaBridge:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def generate(self, model: str, prompt: str, system: Optional[str] = None) -> str:
        """Generate response from Ollama model"""
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        if system:
            data["system"] = system
            
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"Error: {str(e)}"
    
    def available_models(self) -> list:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            return []

def main():
    if len(sys.argv) < 3:
        print("Usage: python claude_ollama_bridge.py <model> <prompt> [system_prompt]")
        sys.exit(1)
    
    model = sys.argv[1]
    prompt = sys.argv[2]
    system = sys.argv[3] if len(sys.argv) > 3 else None
    
    bridge = OllamaBridge()
    response = bridge.generate(model, prompt, system)
    print(response)

if __name__ == "__main__":
    main()