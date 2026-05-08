import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

## Streamlit UI
st.title("Ask anything")

with st.sidebar:
    st.title("Provide OpenAI API Key")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
if not OPENAI_API_KEY:
    st.info("Enter OpenAI API Key to continue")
    st.stop()

# Initialize LLM
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

# Input box
question = st.text_input("Enter question")

# Main Logic. Run only if user enters question
if question:
    response = llm.invoke(question)
    st.write(response.content)

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s13_deployment
# python -m streamlit run streamlit_demo.py
# http://localhost:8501/  
#
# Run from cwd to auto-generate 'requirements.txt'
# pip freeze > requirements.txt
# pip show streamlit

## Deploy to streamlit.app
# create github repo 'my-streamlit-app'
# Push 'streamlit_demo.py' and 'requirements.txt' to repo
# Create account in streamlit.app
# Connect GitHub, browse repo, deploy