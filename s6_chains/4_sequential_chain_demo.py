import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda

## API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

## LLMs
# Ollama cloud LLM's
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
    input_variables=["title", "emotion"],
    template="""
    You are an experienced speech writer.

    Write a powerful {emotion} speech of around 350 words for the following title:

    {title}

    Return only valid JSON.

    The JSON must have exactly these 2 keys:
    {{
    "title": "{title}",
    "speech": "your full speech here"
    }}

    Rules:
    - Do not include emotion as a key.
    - Do not include markdown.
    - Do not include explanation.
    - Do not wrap the JSON in ```json.
    """
)

## LCEL chains
# Helper functions
def display_title(title: str) -> str:
    st.subheader("Generated Title")
    st.markdown(title)
    return title

def title_to_dictionary(title: str) -> dict[str, str]:
    return {
        "title": title,
        "emotion": emotion
    }

# first_chain: Takes topic input, generates a speech title, displays it, and returns the title
first_chain = (
    title_prompt
    | title_llm
    | StrOutputParser()
    # Uncomment below line to display title before speech is generated
    # | RunnableLambda(display_title)         # RunnableLambda(lambda title: (st.write("Generated Title: "), st.write(title), title)[2])
)

# second_chain: Takes {"title": title, "emotion": emotion} and returns parsed JSON with title and speech
second_chain = (
    speech_prompt
    | speech_llm
    | JsonOutputParser()
)

# final_chain: topic -> title -> {"title": title, "emotion": emotion} -> JSON speech output
final_chain = (
    first_chain
    | RunnableLambda(lambda title: {
        "title": title,
        "emotion": emotion
    })
    | second_chain
)

## Streamlit UI
st.title("Speech Generator")

topic = st.text_input("Enter the topic:")
emotion = st.text_input("Enter the emotion:")

## Generate
if st.button("Generate Speech"):
    if topic and emotion:
        with st.spinner("Generating speech..."):
            response = final_chain.invoke({
                "topic": topic
            })

        st.subheader("Generated Title")
        st.write(response["title"])

        st.subheader("Generated Speech")
        st.markdown(response["speech"])

        # Optional debug
        with st.expander("View JSON response"):
            st.json(response)
    else:
        st.warning("Please enter both topic and emotion.")

## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s6_chains
# python -m streamlit run 4_sequential_chain_demo.py
# http://localhost:8501/  
