import os
import asyncio

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

HTTP_URL = "http://localhost:8000/mcp"

async def main():
    # Connect over streamable HTTP
    client = MultiServerMCPClient({
        "demo": {"url": HTTP_URL, "transport": "streamable-http"}
    })

    # 1) Get bio from the resource
    blobs = await client.get_resources(server_name="demo",
                                       uris="docs://aboutme")

    bio_text = blobs[0].as_string() if blobs else ""
    print("Bio:", bio_text[:120], "...")

    # 2) Build prompt messages using the bio as context
    messages = await client.get_prompt(server_name="demo",
                                       prompt_name="question",
                                       arguments={
                                           "question": "What subjects does Bharath Teach",
                                           "context": bio_text
                                       })

    # 3) Send to LLM
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

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s15_mcp
# python 5_resource_prompt_client.py  