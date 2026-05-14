import os
import asyncio
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# Create MCP client using stdio transport
# This client will start the MCP server process itself
client_stdio = MultiServerMCPClient({
    "tools": {
        "command": "python",
        "args": ["1_mcp_server.py"],
        "transport": "stdio"
    }
})

# Fetch tools exposed by the MCP server
# Example: wikipedia_search, ddg_search
tools = asyncio.run(client_stdio.get_tools())

# Create Ollama chat model using Ollama Cloud
llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

# Create an agent using the LLM and MCP tools
agent = create_agent(llm, tools)

# Streamlit page title
st.title("AI Agent (MCP Version)")

# Input box for user task/question
task = st.text_input("Assign me a task")

# Button to trigger agent
ask_button = st.button("Ask")

# Run agent only when button is clicked and task is not empty
if ask_button and task:
    # Send task to the agent
    response = asyncio.run(agent.ainvoke({"messages": task}))

    # Show full response object for debugging
    st.write(response)

    # Extract final assistant answer
    final_output = response["messages"][-1].content

    # Show final answer
    st.write(final_output)

# Run
# cd D:\dev\github\agentic-ai-langchaindemo\s15_mcp
# python -m streamlit run 3_mcp_client_stdio.py
#
# Ask questions from questions.txt file

## Force to use Wikipedia or DDG search
# Q: Use wikipedia_search tool. Who is the author of Pride and Prejudice?
# Q: Use ddg_search tool. Search latest news about OpenAI.