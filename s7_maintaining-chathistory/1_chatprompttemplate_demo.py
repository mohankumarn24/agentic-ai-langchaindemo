import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## 1. OpenAI Cloud API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

## 2. Ollama cloud API key
llm_ollamaCloud = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

## 3. Ollama local
llm_ollamaLocal = OllamaLLM(model="tinyllama")

# PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an Agile Coach. "
            "Answer questions related to Agile process clearly."
        ),
        (
            "human",
            "{input}"
        )
    ]
)

# STREAMLIT UI
st.title("Agile Guide")

input = st.text_input("Enter the question:")

# LCEL CHAIN
chain = prompt_template | llm_ollamaLocal | StrOutputParser()
# chain = prompt_template | llm_ollamaCloud | StrOutputParser()

if input:
    response = chain.invoke({
        "input": input
    })
    st.write(response)

## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s7_maintaining-chathistory
# python -m streamlit run 1_chatprompttemplate_demo.py
# http://localhost:8501/      