import os
import asyncio

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

HTTP_URL = "http://localhost:8000/mcp"

async def main():
    client = MultiServerMCPClient({
        "demo": {
            "url": HTTP_URL,
            "transport": "streamable-http"
        }
    })

    blobs = await client.get_resources(
        server_name="demo",
        uris="docs://aboutme"
    )

    bio_text = blobs[0].as_string() if blobs else ""

    print("Bio:", bio_text[:120], "...")

    messages = await client.get_prompt(
        server_name="demo",
        prompt_name="question",
        arguments={
            "question": "What subjects does Bharath Teach?",
            "context": bio_text
        }
    )

    llm = ChatOllama(
        model="gpt-oss:20b",
        base_url="https://ollama.com",
        headers={
            "Authorization":
                f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
        }
    )
    resp = await llm.ainvoke(messages)
    print("\nLLM Answer:\n", resp.content)

if __name__ == "__main__":
    asyncio.run(main())
