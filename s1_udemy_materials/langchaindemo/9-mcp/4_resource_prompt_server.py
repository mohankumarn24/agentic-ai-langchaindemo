from mcp.server.fastmcp import FastMCP

mcp = FastMCP("promptandresource-mcp-demo")

@mcp.resource("docs://aboutme")
def bharath_bio() -> str:
    return (
        "Bharath Thippireddy is a popular Udemy tech instructor and software architect "
        "with 20+ years of experience in India and the USA. He teaches Java, Python, GenAI, "
        "LangChain, and GitHub Copilot, and builds AI apps (RAG, agents). He runs Neyah Digital Solutions, and works on "
        "ed‑tech and gov-tech ideas in India. He’s also a certified yoga teacher ,actor and an active "
        "content creator on YouTube and LinkedIn."
    )

@mcp.prompt("question")
def ask_about_bharath(question: str, context: str) -> str:
    return (
        "System: You are a helpful assistant. Answer strictly using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"User question:\n{question}\n\n"
        "Answer:"
    )

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
