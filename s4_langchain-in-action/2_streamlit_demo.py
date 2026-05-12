import os
import streamlit as st
import logging

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

# from langchain_core.globals import set_debug
# set_debug(True)

## Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,  # change to INFO in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

## LLM
# OpenAI cloud api
# export/setx OPENAI_API_KEY="your_key"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_llm_cloud = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

# Ollama cloud api
ollama_llm_local = OllamaLLM(model="tinyllama")
ollama_llm_cloud = OllamaLLM(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={                                                # Adds API key to request headers. Ex: Authorization: Bearer xxxxx
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

## Streamlit UI
st.title("Ask anything")
question = st.text_input("Enter the question:")

## Generate
if question:
    logger.debug(f"Received question: {question}")
    try:
        with st.spinner("Thinking..."):
            response = ollama_llm_cloud.invoke(
                f"Answer in 1-2 lines clearly: {question}"
            )

        logger.debug(f"LLM response: {response}")
        st.success("Response:")
        st.write(response)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        st.error("Something went wrong. Check logs.")

## Run
# 1. Tab 1 (PowerShell): 
#    cd D:\dev\github\agentic-ai-langchaindemo
# 
#    # create venv (only first time)
#    python -m venv venv311
# 
#    # allow scripts (only per session)
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# 
#    # activate venv
#    venv311\Scripts\activate
# 
#    # install deps (only first time)
#    pip install langchain langchain-openai langchain-ollama
# 
#    # start ollama server
#    ollama serve
# 
# 2. Tab 2 (PowerShell):
#    cd D:\dev\github\agentic-ai-langchaindemo
#    venv311\Scripts\activate
# 
#    cd D:\dev\github\agentic-ai-langchaindemo\s4_langchain-in-action
# 
#    python -m streamlit run 2_streamlit_demo.py
#    # python -m streamlit run 2_streamlit_demo.py --logger.level=debug
#    # python -m streamlit run 2_streamlit_demo.py --logger.level=info
# 
# 3. UI:
#    http://localhost:8501/  

