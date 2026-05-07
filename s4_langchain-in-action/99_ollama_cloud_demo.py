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