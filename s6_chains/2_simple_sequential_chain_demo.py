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
# OpenAI cloud LLM
llm_openai_cloud = ChatOpenAI(
    model="gpt-5-nano", 
    api_key=openai_api_key, 
    temperature=0                                           # Controls randomness: 0 = deterministic/focused, higher values = more creative/random
)

# Ollama cloud LLM
llm_ollama_cloud = OllamaLLM(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        # Adds API key to request headers. Example: Authorization: Bearer xxxxx
        "Authorization": f"Bearer {ollama_api_key}"
    },
    temperature=0
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

## Streamlit UI
st.title("Speech Generator")

# Select LLM model
selected_provider = st.selectbox(
    "Choose LLM provider",
    options=["OpenAI Cloud", "Ollama Cloud"],
    index=0                                                 # Default: OpenAI Cloud. Use index=1 to default to Ollama Cloud
)

# input fields
topic = st.text_input("Enter topic: ")

# Select LLM based on user selection
if selected_provider == "OpenAI Cloud":
    llm_selected = llm_openai_cloud
else:
    llm_selected = llm_ollama_cloud

## CHAINS (LCEL)
# Step 1: topic -> title_prompt -> LLM -> title -> display title -> return title
# Step 2: title -> speech_prompt -> LLM -> speech
# Step 3: 
#         final_chain = first_chain | second_chain
#         topic -> title -> speech

# Helper functions
def display_title_and_return_title(title: str) -> str:
    st.subheader("Generated Title")
    st.markdown(title)
    return title

first_chain = (
    title_prompt
    | llm_selected
    | StrOutputParser()                                         # Converts LLM response to plain string. See 'Note 1'
                                                                # StrOutputParser() converts the LLM response into a plain Python string
                                                                # AIMessage(content="This is the answer") -> "This is the answer"

    | RunnableLambda(display_title_and_return_title)            # Displays title and passes it to next chain. 
                                                                # Other approaches:  
                                                                #   first_chain = title_prompt | llm_selected | StrOutputParser() | display_title_and_return_title  
                                                                #   first_chain = title_prompt | llm_selected | StrOutputParser() | (lambda title: (st.write(title), title)[1])
                                                                #   first_chain = title_prompt | llm_selected | StrOutputParser() | (lambda title: (st.write("Generated Title"), st.write(title), title)[2])
)

second_chain = (
    speech_prompt
    | llm_selected
    | StrOutputParser()
)

final_chain = first_chain | second_chain

## Generate
if st.button("Generate Speech"):
    if topic:
        with st.spinner(f"Thinking using {selected_provider}..."):
            response = final_chain.invoke({
                "topic": topic
            })
            
        # ChatOpenAI usually returns AIMessage, OllamaLLM usually returns string 
        st.subheader("Generated Speech")                        
        st.markdown(response)                                   # Second chain ends with 'StrOutputParser()'. So, response will already be a plain string
                                                                # Use st.markdown() for nicely formatted text output
                                                                # st.markdown(response) will show headings, bold text, and bullets properly
                                                                # Use st.write()    for normal/basic output
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