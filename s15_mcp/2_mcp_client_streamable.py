import os
import asyncio
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# Create MCP client and connect to the MCP tool server
client_streamable = MultiServerMCPClient({
    "tools": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable-http"
    }
})

# Fetch tools exposed by the MCP server
# Example: wikipedia_search, ddg_search
tools = asyncio.run(client_streamable.get_tools())

# Create Ollama chat model using Ollama Cloud
llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

# Create an agent with the LLM and MCP tools
agent = create_agent(llm, tools)

# Streamlit page title
st.title("AI Agent (MCP Version)")

# Text input for user task/question
task = st.text_input("Assign me a task")

# Button to trigger agent
ask_button = st.button("Ask")

# Run agent only when button is clicked and task is not empty
if ask_button and task:
    # Send user task to the agent
    response = asyncio.run(agent.ainvoke({"messages": task}))
    # Better structured message format
    # response = asyncio.run(agent.ainvoke({
    #     "messages": [{"role": "user", "content": task}]
    # }))
    
    # Show full raw response for debugging
    st.write(response)

    # Extract final assistant answer from response messages
    final_output = response["messages"][-1].content

    # Show final answer to user
    st.write(final_output)




## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s15_mcp
# python -m streamlit run 2_mcp_client_streamable.py
#
# Ask questions from questions.txt file

## Force to use Wikipedia or DDG search
# Q: Use wikipedia_search tool. Who is the author of Pride and Prejudice?
# Q: Use ddg_search tool. Search latest news about OpenAI.