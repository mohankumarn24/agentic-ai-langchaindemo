from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("promptandresource-mcp-demo")

# Register a resource with URI docs://aboutme
# Resource = data/content exposed by the MCP server
@mcp.resource("docs://aboutme")
def bharath_bio() -> str:
    # Return static bio text as resource content
    return (
        "Bharath Thippireddy is a popular Udemy tech instructor and software architect "
        "with 20+ years of experience in India and the USA. He teaches Java, Python, GenAI, "
        "LangChain, and GitHub Copilot, and builds AI apps (RAG, agents). He runs Neyah Digital Solutions, and works on "
        "ed‑tech and gov-tech ideas in India. He’s also a certified yoga teacher ,actor and an active "
        "content creator on YouTube and LinkedIn."
    )

# Register a reusable prompt template named "question"
# Prompt = instruction template exposed by the MCP server
@mcp.prompt("question")
def ask_about_bharath(question: str, context: str) -> str:
    # Build prompt using user question + provided context
    return (
        "System: You are a helpful assistant. Answer strictly using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"User question:\n{question}\n\n"
        "Answer:"
    )

# Run server only when this file is executed directly
if __name__ == "__main__":
    # Start MCP server using streamable HTTP transport
    mcp.run(transport="streamable-http")

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s15_mcp
# python 4_resource_prompt_server.py


# 1. MCP server starts
#         ↓
# 2. MCP server exposes:
#         - Resource: docs://aboutme
#         - Prompt: question
#         ↓
# 3. MCP client connects to MCP server
#         ↓
# 4. Client fetches resource docs://aboutme
#         ↓
# 5. Server returns Bharath bio text
#         ↓
# 6. Client sends:
#         - question
#         - bio_text as context
#    to MCP prompt named "question"
#         ↓
# 7. Server builds prompt/messages using prompt template
#         ↓
# 8. Client receives final prompt messages
#         ↓
# 9. Client sends those messages to LLM
#         ↓
# 10. LLM answers using the provided context