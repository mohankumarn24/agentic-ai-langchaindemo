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

## PromptTemplate: LangChain builds the prompt using variables
prompt_template = PromptTemplate(
    input_variables=["city", "month", "language", "budget"],
    template="""
    You are a helpful travel guide.

    Create a travel guide for {city} for someone visiting in {month}.

    Include:
    1. Must-visit attractions.
    2. Local cuisine the traveler should try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.

    Keep the answer practical, clear, and beginner-friendly.
    """
)

## Streamlit UI
st.title("Travel Guide")

city = st.text_input("Enter city")
month = st.text_input("Enter month of travel")
language = st.text_input("Enter language")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])

## Generate
if city and month and language and budget:
    response = ollama_llm_cloud.invoke(prompt_template.format(city=city,
                                                              month=month,
                                                              language=language,
                                                              budget=budget))
    st.write(response)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s5_prompttemplate
# python -m streamlit run 2_travelguide_demo.py
# http://localhost:8501/  