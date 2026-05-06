import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# OpenAI LLM
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# Ollama LLM
# NOTE: tinyllama is unreliable for strict JSON outputs; use GPT-4o for structured responses
llm = OllamaLLM(model="tinyllama")

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
    You need to write a powerful {emotion} speech of 350 words
    for the following title: {title}
    Format the output with 2 keys: 'title', 'speech' and fill them with respective values
    """
)

# CHAINS (LCEL)
first_chain  = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = speech_prompt | llm | JsonOutputParser()
final_chain  = first_chain | (lambda title:{"title": title,"emotion": emotion}) | second_chain

# Streamlit UI
st.title("Speech Generator")

topic = st.text_input("Enter the topic:")
emotion = st.text_input("Enter the emotion:")

# Generate
if topic and emotion:
    response = final_chain.invoke({"topic":topic})
    st.write(response)
    # st.write(response['title'])

## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s6_chains
# python -m streamlit run 4_sequential_chain_demo.py
# http://localhost:8501/  
