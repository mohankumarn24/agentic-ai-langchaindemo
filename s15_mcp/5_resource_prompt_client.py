import os
import asyncio

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient

# MCP server endpoint
HTTP_URL = "http://localhost:8000/mcp"

async def main():
    # Create MCP client and connect to MCP server over streamable HTTP
    client = MultiServerMCPClient({
        "demo": {
            "url": HTTP_URL,
            "transport": "streamable-http"
        }
    })

    # 1) Fetch bio from MCP server
    # Resource URI is registered in server as docs://aboutme    
    blobs = await client.get_resources(
        server_name="demo",
        uris="docs://aboutme"
    )

    # Convert resource blob to plain string
    bio_text = blobs[0].as_string() if blobs else ""

    # Print first 120 characters for quick verification
    print("Bio:", bio_text[:120], "...")

    # 2) Fetch prompt template from MCP server and build prompt messages using the bio as context
    # The server has prompt registered as @mcp.prompt("question")
    # We pass question and context as arguments to build final messages
    messages = await client.get_prompt(
        server_name="demo",
        prompt_name="question",
        arguments={
            "question": "What subjects does Bharath Teach?",
            "context": bio_text
        }
    )

    # 3) Send to LLM
    llm = ChatOllama(
        model="gpt-oss:20b",
        base_url="https://ollama.com",
        headers={
            "Authorization":
                f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
        }
    )
    # Send prompt messages to LLM
    resp = await llm.ainvoke(messages)
    # Print final LLM answer
    print("\nLLM Answer:\n", resp.content)

# Run async main function when file is executed directly
if __name__ == "__main__":
    asyncio.run(main())



## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s15_mcp
# python 5_resource_prompt_client.py  


# =========================================================
# FLOW
# =========================================================
# MCP client:
#   -> fetches docs://aboutme resource
#   -> extracts bio text
#   -> passes bio text + question to MCP prompt "question"
#   -> receives final prompt messages
#   -> sends messages to LLM
#   -> prints answer
# =========================================================