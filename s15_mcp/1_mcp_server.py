import wikipedia

from duckduckgo_search import DDGS
from mcp.server.fastmcp import FastMCP

# Create an MCP server to expose tools
mcp = FastMCP(name="Tool Server")

# Register this function as an MCP tool
@mcp.tool()
def wikipedia_search(query: str) -> str:
    try:
        # Get 2-sentence summary from Wikipedia
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        # Return error message if Wikipedia search fails
        return f"Error: {str(e)}"

# Register this function as another MCP tool
@mcp.tool()
def ddg_search(query: str) -> str:
    try:
        # Create DuckDuckGo search client
        with DDGS() as ddgs:
            # Get top 3 text search results
            results = ddgs.text(query, max_results=3)

            # Extract only result body text and join into one string
            return "\n".join([r["body"] for r in results])
    except Exception as e:
        # Return error message if DuckDuckGo search fails
        return f"Error: {str(e)}"

# Run server only when this file is executed directly
if __name__ == "__main__":
    # Start MCP server using HTTP transport
    mcp.run(transport="streamable-http")            # use this when running 2_mcp_client_streamable.py
                                                    # 
                                                    # In the HTTP version, you manually start the server first:
                                                    #   python 1_mcp_server.py
                                                    # 
                                                    # Then run Streamlit separately.

    # Start MCP server using stdio transport
    # mcp.run(transport="stdio")                    # use this when running 3_mcp_client_stdio.py
                                                    # 
                                                    # In the stdio version, the client itself starts the server using:
                                                    #   "command": "python",
                                                    #   "args": ["1_mcp_server.py"]
                                                    #
                                                    # Then run only
                                                    #   python -m streamlit run 3_mcp_client_stdio.py



## Run
# pip install requirements.txt
#
# cd D:\dev\github\agentic-ai-langchaindemo\s15_mcp
# python 1_mcp_server.py
# Uvicorn running on http://127.0.0.1:8000


## Flow:

# 1_mcp_tool_server.py
#         exposes tools:
#         - wikipedia_search
#         - ddg_search
#
#                 ↑ HTTP MCP connection
#
# 2_mcp_client_streamable.py
#         connects to server
#         gets tools
#         gives tools to LLM agent
#         runs Streamlit UI




# User types task in Streamlit
#         ↓
# LangChain agent receives task
#         ↓
# Agent uses gpt-oss:20b through Ollama Cloud
#         ↓
# LLM decides whether a tool is needed
#         ↓
# If needed, agent calls MCP tool
#         ↓
# MCP server runs wikipedia_search/ddg_search
#         ↓
# Tool result comes back to agent
#         ↓
# LLM creates final answer
#         ↓
# Streamlit displays response