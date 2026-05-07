import os
from ollama import Client

client = Client(
    host="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

response = client.embed(
    model="nomic-embed-text",
    input="What is Scrum?"
)

print(response)