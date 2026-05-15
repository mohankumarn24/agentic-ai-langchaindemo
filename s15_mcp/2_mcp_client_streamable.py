import os
import asyncio
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

## API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

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

## LLMs
# OpenAI cloud LLM
llm_openai_cloud = ChatOpenAI(
    model="gpt-5-nano", 
    api_key=openai_api_key, 
    temperature=0                                           # Controls randomness: 0 = deterministic/focused, higher values = more creative/random
)

# Create Ollama chat model using Ollama Cloud
llm_ollama_cloud = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

## Streamlit UI
st.title("AI Agent (MCP Version)")

# Select LLM model
selected_provider = st.selectbox(
    "Choose LLM provider",
    options=["OpenAI Cloud", "Ollama Cloud"],
    index=0                                                 # Default: OpenAI Cloud. Use index=1 to default to Ollama Cloud
)

# Text input for user task/question
task = st.text_input("Assign me a task")

# Select LLM based on user selection
if selected_provider == "OpenAI Cloud":
    llm_selected = llm_openai_cloud
else:
    llm_selected = llm_ollama_cloud

## Agent
# Create an agent with the LLM and MCP tools
agent = create_agent(llm_selected, tools) 

# Button to trigger agent
ask_button = st.button("Ask")

# Run agent only when button is clicked and task is not empty
if ask_button and task:
    with st.spinner(f"Thinking using {selected_provider}..."):
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
# Q: Use ddg_search tool. Search latest news about OpenAI



# Simple flow:
#
# User asks question
#       ↓
# AI agent receives question
#       ↓
# Agent decides whether a tool is needed
#       ↓
# Agent calls MCP tool
#       ↓
# MCP server runs the Python function
#       ↓
# Tool result returns to agent
#       ↓
# Agent gives final answer