import os
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import PromptTemplate

# OpenAI LLM
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# Ollama LLM
llm = OllamaLLM(model="tinyllama")

# Prompt
prompt_template = PromptTemplate(
    input_variables=["country", "no_of_paras", "language"],
    template="""
    You are an expert in traditional cuisines.
    You provide information about a specific dish from a specific country.
    Avoid giving information about fictional places. If the country is fictional
    or non-existent answer: I don't know.
    Answer the question: What is the traditional cuisine of {country}?
    Answer in {no_of_paras} short paragraphs in {language}
    """
)

# Streamlit UI
st.title("Cuisine Info")

country = st.text_input("Enter country:")
no_of_paras = st.number_input("Enter number of paragraphs", 
                              min_value=1, 
                              max_value=5)
language = st.text_input("Enter language:")

# Generate
if country and language:
    response = llm.invoke(prompt_template.format(country=country,
                                                 no_of_paras=no_of_paras,
                                                 language=language
                                                 ))
    st.write(response)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s5_prompttemplate
# python -m streamlit run 1_prompttemplate_demo.py
# http://localhost:8501/  