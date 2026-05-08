import wikipedia
from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Tool Server")

@mcp.tool()
def wikipedia_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def ddg_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            return "\n".join([r["body"] for r in results])
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")            # use this when running 2_mcp_client_streamable.py
    # mcp.run(transport="stdio")                    # use this when running 3_mcp_client_stdio.py

## Run
# pip install requirements.txt
#
# cd D:\dev\github\agentic-ai-langchaindemo\s15_mcp
# python 1_mcp_server.py
# Uvicorn running on http://127.0.0.1:8000