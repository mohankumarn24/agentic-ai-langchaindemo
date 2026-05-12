import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

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

## Prompts
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an experienced speech writer.

    Craft an impactful title for a speech on the following topic:
    {topic}

    Answer exactly with one title.
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""
    Write a powerful speech of around 350 words for the following title:

    {title}
    """
)

## CHAINS (LCEL)
# Step 1: topic -> title_prompt -> LLM -> title -> display title -> return title
# Step 2: title -> speech_prompt -> LLM -> speech
# Step 3: 
#         final_chain = first_chain | second_chain
#         topic -> title -> speech

def display_title_and_return_title(title):
    st.write("Generated Title: ")
    st.write(title)
    return title

first_chain = (
    title_prompt
    | ollama_llm_cloud
    | StrOutputParser()                                         # Converts LLM response to plain string. See 'Note 1'
    | RunnableLambda(display_title_and_return_title)            # Displays title and passes it to next chain. 
                                                                # Other approaches:  
                                                                #   first_chain = title_prompt | ollama_llm_cloud | StrOutputParser() | display_title_and_return_title  
                                                                #   first_chain = title_prompt | ollama_llm_cloud | StrOutputParser() | (lambda title: (st.write(title), title)[1])
                                                                #   first_chain = title_prompt | ollama_llm_cloud | StrOutputParser() | (lambda title: (st.write("Generated Title"), st.write(title), title)[2])
)

second_chain = (
    speech_prompt
    | ollama_llm_cloud
    | StrOutputParser()
)

final_chain  = first_chain   | second_chain

## Streamlit UI
st.title("Speech Generator")
topic = st.text_input("Enter topic: ")

## Generate
if st.button("Generate Speech"):
    if topic:
        response = final_chain.invoke({
            "topic": topic
        })

        st.write("Generated Speech:")                           # Use st.write()    for normal/basic output
        st.markdown(response)                                   # Use st.markdown() for nicely formatted text output
                                                                # st.markdown(response) will show headings, bold text, and bullets properly
    else:
        st.warning("Please enter a topic")


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s6_chains
# python -m streamlit run 2_simple_sequential_chain_demo.py
# http://localhost:8501/  


## Note 1: StrOutputParser()
#
# If you use ChatOpenAI or ChatOllama, the model returns:
#     AIMessage(content="...")
#
# StrOutputParser() converts it to:
#     "..."
#
# Meaning:
#     LLM response object -> plain string
#
# RunnableLambda(display_title_and_return_title):
#     1. Displays the generated title in Streamlit.
#     2. Returns the same title so second_chain can use it.


## Note 2
# PromptTemplate:
#     Creates reusable/dynamic prompts using input variables.
#
# llm:
#     Sends the final prompt to the AI model and returns the model response.
#
# StrOutputParser:
#     Converts the LLM response into a plain string.
#     Useful especially when using ChatOpenAI or ChatOllama,
#     because they return AIMessage(content="...").