import streamlit as st
import logging
from langchain_ollama import OllamaLLM

# from langchain_core.globals import set_debug
# set_debug(True)

# ============================================================
# Logging Configuration
# ============================================================
logging.basicConfig(
    level=logging.DEBUG,  # change to INFO in production
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# Initialize LLM
# ============================================================
ollama_llm = OllamaLLM(model="tinyllama")

# ============================================================
# Streamlit UI
# ============================================================
st.title("Ask anything")
# Input box
question = st.text_input("Enter the question:")

# ============================================================
# Main Logic. Run only if user enters question
# ============================================================
if question:
    logger.debug(f"Received question: {question}")
    try:
        with st.spinner("Thinking..."):
            response = ollama_llm.invoke(
                f"Answer in 1-2 lines clearly: {question}"
            )

        logger.debug(f"LLM response: {response}")
        st.success("Response:")
        st.write(response)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        st.error("Something went wrong. Check logs.")

"""
1. Tab 1 (PowerShell): 
   cd D:\dev\github\agentic-ai-langchaindemo

   # create venv (only first time)
   python -m venv venv311

   # allow scripts (only per session)
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

   # activate venv
   venv311\Scripts\activate

   # install deps (only first time)
   pip install langchain langchain-openai langchain-ollama

   # start ollama server
   ollama serve

2. Tab 2 (PowerShell):
   cd D:\dev\github\agentic-ai-langchaindemo
   venv311\Scripts\activate

   python -m streamlit run streamlit_demo.py
   # python -m streamlit run streamlit_demo.py --logger.level=debug
   # python -m streamlit run streamlit_demo.py --logger.level=info

3. UI:
   http://localhost:8501/  

"""

"""
============================================================
SETUP + RUN STREAMLIT APP (WINDOWS - POWERSHELL)
============================================================

Prerequisite:
- Python 3.11.0 installed
- VS Code / PowerShell

------------------------------------------------------------
STEP 1: Navigate to project folder
------------------------------------------------------------
cd D:\dev\github\agentic-ai-langchaindemo

------------------------------------------------------------
STEP 2: Create virtual environment
------------------------------------------------------------
# Creates isolated Python environment (prevents conflicts)
python -m venv venv311

------------------------------------------------------------
STEP 3: Allow script execution (temporary)
------------------------------------------------------------
# Required because PowerShell blocks scripts by default
# This applies ONLY to current session (safe)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

------------------------------------------------------------
STEP 4: Activate virtual environment
------------------------------------------------------------
# Activates venv → all packages install locally
venv311\Scripts\activate

# You should see:
# (venv311) in terminal

------------------------------------------------------------
STEP 5: Install required packages
------------------------------------------------------------
# Streamlit → UI
# LangChain → LLM framework
# langchain-ollama → local LLM integration
pip install streamlit langchain langchain-ollama

------------------------------------------------------------
STEP 6: Start Ollama (in separate terminal)
------------------------------------------------------------
# Required for local LLM to work
ollama serve

------------------------------------------------------------
STEP 7: Run Streamlit app
------------------------------------------------------------
# Launches web app in browser (http://localhost:8501/)
python -m streamlit run streamlit_demo.py
# python -m streamlit run streamlit_demo.py --logger.level=debug
# python -m streamlit run streamlit_demo.py --logger.level=info  

------------------------------------------------------------
NOTES
------------------------------------------------------------
- Always activate venv before running app
- Use PowerShell (avoid Git Bash for Python apps)
- Ollama must be running before app execution
- First run may take time (model loading)

============================================================
"""
