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

## PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["city", "month", "language", "budget"],
    template="""
    You are a helpful travel guide.

    Create a practical travel guide for {city} for someone visiting in {month}.

    Include:
    1. Must-visit attractions.
    2. Local cuisine the traveler should try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.

    Keep the answer clear, useful, and beginner-friendly.
    """
)

## Streamlit UI
st.title("Travel Guide")

city = st.text_input("Enter city")
month = st.text_input("Enter month of travel")
language = st.text_input("Enter language")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])

## Langchain Expression Language (LCEL)
# '|' means take output from left side and send it to right side
#  ie., first fill the prompt template, then send the final prompt to ollama_llm_cloud
#
# Meaning:
#     1. Take input dictionary
#     2. Fill PromptTemplate placeholders
#     3. Create final prompt string
#     4. Send final prompt to LLM
#     5. Return LLM response
#
# Internally, 'chain.invoke({...})' is basically a shortcut for: 
#     final_prompt = prompt_template.format(city=city, month=month, language=language, budget=budget)
#     response = ollama_llm_cloud.invoke(final_prompt)

chain = prompt_template | ollama_llm_cloud

## Generate
# The LLM is called only when you click the button
if st.button("Generate Travel Guide"):
    if city and month and language and budget:
        response = chain.invoke({
            "city": city,
            "month": month,
            "language": language,
            "budget": budget
        })

        st.success("Response:")
        if hasattr(response, "content"):
            # When using ChatOllama or ChatOpenAI, response may be an AIMessage/chat message object
            # Example:
            #     response -> AIMessage(content="Hello! How can I help you?")
            st.write(response.content)
        else:
            # When using OllamaLLM, response is usually a plain string
            st.write(response)
    else:
        st.warning("Please fill all fields")


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s6_chains
# python -m streamlit run 1_lcel_demo.py
# http://localhost:8501/  