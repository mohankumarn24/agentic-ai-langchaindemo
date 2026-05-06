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
    input_variables=["city", "month", "language", "budget"],
    template="""
    Welcome to the {city} travel guide!
    If you're visiting in {month}, here's what you can do:
    1. Must-visit attractions.
    2. Local cuisine you must try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.
    Enjoy your trip!
    """
)

# Streamlit UI
st.title("Travel Guide")

city = st.text_input("Enter city:")
month = st.text_input("Enter month of travel")
language = st.text_input("Enter language:")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])

# Generate
if city and month and language and budget:
    response = llm.invoke(prompt_template.format(city=city,
                                                 month=month,
                                                 language=language,
                                                 budget=budget
                                                 ))
    st.write(response)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s5_prompttemplate
# python -m streamlit run 2_travelguide_demo.py
# http://localhost:8501/  