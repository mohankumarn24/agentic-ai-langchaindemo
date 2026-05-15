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

## Streamlit UI
st.title("Cuisine Info")

# Select LLM model
selected_provider = st.selectbox(
    "Choose LLM provider",
    options=["OpenAI Cloud", "Ollama Cloud"],
    index=0                                                 # Default: OpenAI Cloud. Use index=1 to default to Ollama Cloud
)

# input fields
country = st.text_input("Enter country")
no_of_paragraphs = st.number_input("Enter number of paragraphs", 
                                   min_value=1, 
                                   max_value=5)
language = st.text_input("Enter language")

# Select LLM based on user selection
if selected_provider == "OpenAI Cloud":
    llm_selected = llm_openai_cloud
else:
    llm_selected = llm_ollama_cloud

## Generate
ask_button = st.button("Ask")
if ask_button and country and language:
    with st.spinner(f"Thinking using {selected_provider}..."):
        response = llm_selected.invoke(
            prompt_template.format(country=country,
                                   no_of_paragraphs=no_of_paragraphs,
                                   language=language))
        
    # st.write("Raw response:")
    # st.write(response)

    # st.write("Response type:")
    # st.write(type(response))

    st.success("Response:")

    # ChatOpenAI usually returns AIMessage, OllamaLLM usually returns string
    # Example: response -> AIMessage(content="Hello! How can I help you?")
    if hasattr(response, "content"):
        st.write(response.content)
    else:
        st.write(response)


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s5_prompttemplate
# python -m streamlit run 1_prompttemplate_demo.py
# http://localhost:8501/  