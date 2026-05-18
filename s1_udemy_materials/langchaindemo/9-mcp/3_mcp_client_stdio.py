import os
import asyncio
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

client_stdio = MultiServerMCPClient({
    "tools": {
        "command": "python",
        "args": ["1_mcp_server.py"],
        "transport": "stdio"
    }
})

tools = asyncio.run(client_stdio.get_tools())

llm_openai_cloud = ChatOpenAI(
    model="gpt-5-nano", 
    api_key=openai_api_key, 
    temperature=0
)

llm_ollama_cloud = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

st.title("AI Agent (MCP Version)")

selected_provider = st.selectbox(
    "Choose LLM provider",
    options=["OpenAI Cloud", "Ollama Cloud"],
    index=0
)

task = st.text_input("Assign me a task")

if selected_provider == "OpenAI Cloud":
    llm_selected = llm_openai_cloud
else:
    llm_selected = llm_ollama_cloud

agent = create_agent(llm_selected, tools)

ask_button = st.button("Ask")
if ask_button and task:
    with st.spinner(f"Thinking using {selected_provider}..."):  
        response = asyncio.run(agent.ainvoke({"messages": task}))
    st.write(response)
    final_output = response["messages"][-1].content
    st.write(final_output)
