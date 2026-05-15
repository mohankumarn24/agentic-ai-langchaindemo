import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

## API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

## LLMs
## Experiment with different LLM combinations:
#     {Gemini, OpenAI}
#     {OpenAI, Ollama local}
#     {Ollama local, Ollama local}
#

# See 'Note'
# Active selection Ollama cloud LLM's
title_llm = OllamaLLM(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization": f"Bearer {ollama_api_key}"
    },
    temperature=0.2
)

speech_llm = OllamaLLM(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization": f"Bearer {ollama_api_key}"
    },
    temperature=0.8
)

# Prompts
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

## LCELs
# Helper functions
def display_title(title: str) -> str:
    st.subheader("Generated Title")
    st.markdown(title)
    return title

# def title_to_dictionary(title):
# def title_to_dictionary(title: str) -> dict:    
def title_to_dictionary(title: str) -> dict[str, str]:    
    return {"title": title}

# first_chain: Takes topic input, generates a speech title, displays it, and returns the title
first_chain = (
    title_prompt
    | title_llm
    | StrOutputParser()                     # StrOutputParser() converts the LLM response into a plain Python string
                                            # AIMessage(content="This is the answer") -> "This is the answer"
    | RunnableLambda(display_title)         # Same as: lambda title: (st.write("Generated Title"), st.write(title), title)[2]
)

# second_chain: Takes the generated title and returns the final speech.
# LCEL chains automatically return the output of their last chain; here, second_chain returns the speech
second_chain = (
    speech_prompt
    | speech_llm
    | StrOutputParser()
)

# final_chain: Connects first_chain and second_chain so topic becomes title, then title becomes speech.
# Runs title generation, converts title to {"title": title}, then generates the speech
final_chain = (
    first_chain
    | RunnableLambda(title_to_dictionary)   # Converts title string -> {"title": title}
    | second_chain
)

## UI
st.title("Speech Generator")
topic = st.text_input("Enter topic")

## Generate
if st.button("Generate Speech"):
    if topic:
        with st.spinner("Generating speech..."):
            response = final_chain.invoke({
                "topic": topic
            })

        st.subheader("Generated Speech")
        st.markdown(response)
    else:
        st.warning("Please enter a topic")    


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s6_chains
# python -m streamlit run 3_multiple_llms_demo.py
# http://localhost:8501/  


## Note
# Try different temperatures to observe output variation.
# Temperature:
#     Lower temperature, e.g. 0.2: More focused, consistent, less creative.
#     Higher temperature, e.g. 0.8: More creative, varied, expressive.
# Suggested use:
#     title generation  -> lower temperature
#     speech generation -> higher temperature

## 1. Different local Ollama models
# title_llm  = OllamaLLM(model="llama3" , temperature=0.2)  # often better for structured reasoning/title generation
# speech_llm = OllamaLLM(model="mistral", temperature=0.8)  # often good for fluent speech generation

## 2. Same local Ollama model with different temperatures
# title_llm  = OllamaLLM(model="tinyllama", temperature=0.2)
# speech_llm = OllamaLLM(model="tinyllama", temperature=0.8)

## 3. OpenAI models with different temperatures
# title_llm  = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
# speech_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

## 4. Mixed LangChain-native setup
# title_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
# speech_llm = OllamaLLM(model="mistral", temperature=0.8)

## 5. Ollama local & Ollama cloud
# title_llm = OllamaLLM(model="tinyllama", temperature=0.2)
# speech_llm = OllamaLLM(
#     model="tinyllama",
#     base_url="https://ollama.com",
#     headers={
#         "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
#     },
#     temperature=0.8
# )