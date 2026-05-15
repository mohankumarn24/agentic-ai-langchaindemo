import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

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

# Select LLM model
selected_provider = st.selectbox(
    "Choose LLM provider",
    options=["OpenAI Cloud", "Ollama Cloud"],
    index=0                                                 # Default: OpenAI Cloud. Use index=1 to default to Ollama Cloud
)

# input fields
city = st.text_input("Enter city")
month = st.text_input("Enter month of travel")
language = st.text_input("Enter language")
budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"])

# Select LLM based on user selection
if selected_provider == "OpenAI Cloud":
    llm_selected = llm_openai_cloud
else:
    llm_selected = llm_ollama_cloud

## Generate
ask_button = st.button("Ask")
if ask_button and city and month and language and budget:
    with st.spinner(f"Thinking using {selected_provider}..."):
        response = llm_selected.invoke(
            prompt_template.format(
                city=city,
                month=month,
                language=language,
                budget=budget))
    
    # st.write("Raw response:")
    # st.write(response)

    # st.write("Response type:")
    # st.write(type(response))

    st.success("Response:")

    # ChatOpenAI usually returns AIMessage, OllamaLLM usually returns string
    if hasattr(response, "content"):
        st.write(response.content)
    else:
        st.write(response)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s5_prompttemplate
# python -m streamlit run 2_travelguide_demo.py
# http://localhost:8501/  