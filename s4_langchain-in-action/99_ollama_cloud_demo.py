import os
from ollama import Client

client = Client(
    host="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

response = client.chat(
    model="gpt-oss:120b",
    messages=[
        {
            "role": "user",
            "content": "Write a speech on discipline"
        }
    ]
)

print(response['message']['content'])

# -------------------------------------------------
# RUN
# -------------------------------------------------
# Windows CMD:
# set OLLAMA_API_KEY=your_api_key
#
# PowerShell:
# $env:OLLAMA_API_KEY="your_api_key"
#
# Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s4_langchain-in-action
# python 99_ollama_cloud_demo.py



"""
OllamaLLM vs Client - Recommendation Summary

| Topic                          | OllamaLLM                              | Client                              |
|--------------------------------|---------------------------------------|-------------------------------------|
| Package                        | langchain_ollama                      | ollama                              |
| Purpose                        | LangChain wrapper for Ollama          | Direct Ollama API client            |
| Best used for                  | LangChain chains, agents, RAG         | Direct testing or raw API calls     |
| Common methods                 | invoke()                              | chat(), generate(), embeddings()    |
| Works naturally with LangChain | Yes                                   | No                                  |
| Default target                 | Local Ollama server                   | Local or remote Ollama host         |
| Typical local host             | http://localhost:11434                | http://localhost:11434              |
| Cloud/API key usage            | Not always straightforward            | Easier using host + headers         |
| Recommended for RAG learning   | Yes                                   | Only for testing Ollama directly    |
| Example use case               | Retriever -> Prompt -> OllamaLLM      | Python -> Ollama API -> response    |

Recommended choice:
    Use OllamaLLM when building LangChain RAG examples.

Simple rule:
    If using LangChain, use OllamaLLM.
    If calling Ollama directly, use Client.
"""