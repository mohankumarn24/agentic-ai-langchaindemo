import os
import streamlit as st
import logging

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

# from langchain_core.globals import set_debug
# set_debug(True)

## Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,                                   # change to INFO in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

## Streamlit UI
st.title("Ask anything")

# Select LLM model
selected_provider = st.selectbox(
    "Choose LLM provider",
    options=["OpenAI Cloud", "Ollama Cloud"],
    index=0                                                 # Default: 'OpenAI Cloud'. if 'index=1' -> 'Ollama Cloud'
)

question = st.text_input("Enter question")

ask_button = st.button("Ask")

## Select LLM based on user selection
if selected_provider == "OpenAI Cloud":
    llm_selected = llm_openai_cloud
else:
    llm_selected = llm_ollama_cloud

## Generate
if ask_button and question:
    logger.debug(f"Selected provider: {selected_provider}")
    logger.debug(f"Received question: {question}")

    try:
        with st.spinner(f"Thinking using {selected_provider}..."):
            response = llm_selected.invoke(
                f"Answer in 1-2 lines clearly: {question}"
            )

        # st.write("Raw response:")
        # st.write(response)

        # st.write("Response type:")
        # st.write(type(response))

        logger.debug(f"LLM response: {response}")

        st.success("Response:")

        # ChatOpenAI usually returns AIMessage, OllamaLLM usually returns string
        if hasattr(response, "content"):
            st.write(response.content)
        else:
            st.write(response)

    except Exception as e:
        logger.exception("Error occurred")
        st.error("Something went wrong.")
        st.exception(e)

## Run
#  cd D:\dev\github\agentic-ai-langchaindemo\s4_langchain-in-action
# 
#  python -m streamlit run 2_streamlit_demo.py
#  # python -m streamlit run 2_streamlit_demo.py --logger.level=debug
#  # python -m streamlit run 2_streamlit_demo.py --logger.level=info
# 
#  UI:
#    http://localhost:8501/  

