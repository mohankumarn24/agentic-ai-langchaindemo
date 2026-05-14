import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.globals import set_debug

set_debug(True)

## This is a simple AI Agent app using:
#  - Ollama cloud model
#  - LangChain v1 agent
#  - Tools like Wikipedia and DuckDuckGo
#  - Streamlit UI

# A normal chatbot only answers from its own model knowledge.

# An agent can decide:
#     “Do I need to search Wikipedia or DuckDuckGo before answering?”
# Then it calls tools, reads results, and gives the final answer.


# ------------------------------
# 1. LLM setup
# ------------------------------
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Ollama LLM
# llm = OllamaLLM(model="tinyllama")

if not os.environ.get("OLLAMA_API_KEY"):
    st.error("OLLAMA_API_KEY is missing.")
    st.stop()

llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

# ---------------------------------
# 2. Tools (Wikipedia + DuckDuckGo)
# ---------------------------------
# What is a tool?
#   A tool is an external capability given to the LLM.
#   
#   Example:
#   1. LLM alone:
#       "What is current news?" → may not know
#
#   2. LLM with search tool:
#       "I should search DuckDuckGo" → calls ddg-search → reads result → answers

tools = load_tools(["wikipedia", "ddg-search"])

# ------------------------------
# 3. ReAct-style system prompt
# ------------------------------
# ReAct = Reasoning + Acting
#   Reasoning = think what to do
#   Acting = call tools
# 
# The agent follows this pattern:
#   Thought → Action → Observation → Thought → Action → Observation → Final Answer
# 
# Example:
#   User asks:
#   Who is the current CEO of Microsoft?

# Agent internally may do:
#   Thought: This may require current information.
#   Action: Use ddg-search.
#   Observation: Search result says Satya Nadella is CEO.
#   Thought: I have enough information.
#   Final Answer: The current CEO of Microsoft is Satya Nadella.

tool_names = ", ".join([tool.name for tool in tools])
react_system_prompt = """
                      You are a helpful ReAct-style AI agent.
					  
                      You have access to these tools:
                      {tool_names}

                      Use tools only when needed.
                      Use Wikipedia for encyclopedia-style facts.
                      Use search for current or web-based information.

                      If you use a tool, base your final answer on the tool result.
                      Give a clear final answer to the user.
                      """

# ------------------------------
# 4. Create Agent (new v1 API)
# ------------------------------
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt
)

# ------------------------------
# 5. Streamlit UI
# ------------------------------
st.title("AI Agent (ReAct style - LangChain v1)")
task = st.text_input("Assign me a task")

## Uncomment if you need to enable history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

if st.button("Run Agent") and task:
    ## Uncomment if you need to enable history
    # st.session_state.messages.append({
    #     "role": "user",
    #     "content": task
    # })

    with st.spinner("Agent is working..."):
        try:
            result = agent.invoke(
                {
                    "messages": [{
                        "role": "user",
                        "content": task
                    }]
                }
            )

            final_msg = result["messages"][-1]
            st.write(final_msg.content)
        except Exception as e:
            st.error("Agent failed while running a tool.")
            st.exception(e)


        ## Uncomment if you need to enable history
        # st.session_state.messages.append({
        #     "role": "assistant",
        #     "content": final_msg.content
        # })

## Uncomment if you need to enable history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])




## Run
# pip install wikipedia
# pip install duckduckgo-search 
# pip install -U ddgs
#
# cd D:\dev\github\agentic-ai-langchaindemo\s12_agents
# python -m streamlit run 1_agent_demo.py
# http://localhost:8501/  




# User enters task in Streamlit
#         ↓
# LangChain agent receives the user message
#         ↓
# LLM thinks what to do
#         ↓
# If needed, it calls Wikipedia / DuckDuckGo
#         ↓
# Tool returns observation/result
#         ↓
# LLM uses that result
#         ↓
# Final answer is shown in Streamlit