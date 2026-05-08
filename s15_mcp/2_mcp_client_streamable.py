import os
import asyncio
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client_streamable = MultiServerMCPClient({
    "tools": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable-http"
    }
})

tools = asyncio.run(client_streamable.get_tools())

llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)
agent = create_agent(llm, tools)

st.title("AI Agent (MCP Version)")
task = st.text_input("Assign me a task")

if task:
    response = asyncio.run(agent.ainvoke({"messages": task}))
    # response = asyncio.run(agent.ainvoke({"messages": [{"role": "user","content": task}]}))
    st.write(response)
    final_output = response["messages"][-1].content
    st.write(final_output)

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s15_mcp
# python -m streamlit run 2_mcp_client_streamable.py
#
# Ask questions from questions.txt file