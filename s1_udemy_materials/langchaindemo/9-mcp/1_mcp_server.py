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
    mcp.run(transport="streamable-http")
    # mcp.run(transport="stdio")