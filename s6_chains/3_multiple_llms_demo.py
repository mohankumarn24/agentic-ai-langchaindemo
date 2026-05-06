import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# OpenAI LLM
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# Ollama LLM
llm1 = OllamaLLM(model="tinyllama", temperature=0.2)
llm2 = OllamaLLM(model="tinyllama", temperature=0.8)

## Experiment with different LLM combinations:
## {Gemini, OpenAI}, {OpenAI, Ollama}, {Ollama, Ollama}
## Try different temperatures to observe output variation

# llm1 = OllamaLLM(model="llama3")    # often better for structured reasoning (title generation)
# llm2 = OllamaLLM(model="mistral")   # often faster and more fluent (speech generation)

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
title_chain = title_prompt | llm1 | StrOutputParser()
speech_chain = speech_prompt | llm2 | StrOutputParser()

# UI
st.title("Speech Generator")

topic = st.text_input("Enter topic:")

if topic:
    with st.spinner("Generating title..."):
        title = title_chain.invoke({"topic": topic})

    st.subheader("Title")
    st.write(title)

    with st.spinner("Generating speech..."):
        speech = speech_chain.invoke({"title": title})

    st.subheader("Speech")
    st.write(speech)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s6_chains
# python -m streamlit run 3_multiple_llms_demo.py
# http://localhost:8501/  
