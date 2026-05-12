import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

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

# PromptTemplate: LangChain builds the prompt using variables
prompt_template = PromptTemplate(
    input_variables=["country", "no_of_paragraphs", "language"],
    template="""
    You are an expert in traditional cuisines.

    If the country is fictional or non-existent, answer exactly:
    I don't know.

    Question:
    What is the traditional cuisine of {country}?

    Instructions:
    - Answer in {no_of_paragraphs} short paragraphs.
    - Use {language}.
    - Keep the answer factual and concise.
    """
)

# Streamlit UI
st.title("Cuisine Info")

country = st.text_input("Enter country")
no_of_paragraphs = st.number_input("Enter number of paragraphs", 
                                   min_value=1, 
                                   max_value=5)
language = st.text_input("Enter language:")

# Generate
if country and language:
    
    with st.spinner("Thinking..."):
        response = ollama_llm_cloud.invoke(prompt_template.format(country=country,
                                                                no_of_paragraphs=no_of_paragraphs,
                                                                language=language
                                                                ))
    st.success("Response: ")
    st.write(response)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s5_prompttemplate
# python -m streamlit run 1_prompttemplate_demo.py
# http://localhost:8501/  