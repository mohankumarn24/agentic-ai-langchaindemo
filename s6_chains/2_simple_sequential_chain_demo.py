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
    You need to write a powerful speech of 350 words
    for the following title: {title}
    """
)

## Alternative instead of '(lambda title: (st.write(title), title)[1])'. Currently unused
# Display generated title and return it
# first_chain = title_prompt | llm | StrOutputParser() | display_title
# or
# first_chain = (
#    title_prompt
#    | llm
#    | StrOutputParser()
#    | display_title
# )
def display_title(title):
    st.write("### Generated Title")
    st.write(title)
    return title

# CHAINS (LCEL)
# Step 1: Generate title and display it             : topic -> title prompt -> LLM -> string output -> display title
# Step 2: Generate speech using generated title     : title -> speech prompt -> LLM -> speech
# Step 3: Create sequential AI pipeline             : topic -> generated title -> generated speech
# first_chain  = title_prompt  | llm | StrOutputParser() | (lambda title: (st.write("### Generated Title"), st.write(title), title)[1])
first_chain  = title_prompt  | llm | StrOutputParser() | (lambda title: (st.write(title), title)[1])
second_chain = speech_prompt | llm | StrOutputParser()
final_chain  = first_chain | second_chain

# Streamlit UI
st.title("Speech Generator")

topic = st.text_input("Enter topic:")

# Generate
if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s6_chains
# python -m streamlit run 2_simple_sequential_chain_demo.py
# http://localhost:8501/  

## Notes
# PromptTemplate   = creates dynamic prompts using input variables
# llm              = sends prompt to AI model and generates response
# StrOutputParser  = converts LLM response object into plain string