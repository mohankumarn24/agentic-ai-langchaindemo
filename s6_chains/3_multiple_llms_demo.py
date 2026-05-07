import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# OpenAI LLM
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

## Experiment with different LLM combinations:
## {Gemini, OpenAI}, {OpenAI, Ollama}, {Ollama, Ollama}
## Try different temperatures to observe output variation
# llm1 = OllamaLLM(model="llama3")    # often better for structured reasoning (title generation)
# llm2 = OllamaLLM(model="mistral")   # often faster and more fluent (speech generation)

# 1. Ollama local LLM's
# llm1 = OllamaLLM(model="tinyllama", temperature=0.2)
# llm2 = OllamaLLM(model="tinyllama", temperature=0.8)

# 2. Ollama local & Ollama cloud
llm_ollamaLocal = OllamaLLM(model="tinyllama", temperature=0.2)
llm_ollamaCloud = OllamaLLM(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

# Prompts
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title.	
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""
    You need to write a powerful speech of 350 words
    for the following title: {title}
    """
)

# Chains (PURE LOGIC ONLY)
first_chain = (
    title_prompt
    | llm_ollamaLocal
    | StrOutputParser()
    | (lambda title: (st.write("Generated Title:"), st.write(title), title)[2])
)

second_chain = (
    speech_prompt
    | llm_ollamaCloud
    | StrOutputParser()
)

final_chain = (
    first_chain
    | (lambda title: {"title": title})
    | second_chain
)

# UI
st.title("Speech Generator")

topic = st.text_input("Enter topic:")

if topic:
    response = final_chain.invoke({
        "topic": topic
    })
    st.subheader("Generated Speech")
    st.write(response)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s6_chains
# python -m streamlit run 3_multiple_llms_demo.py
# http://localhost:8501/  
