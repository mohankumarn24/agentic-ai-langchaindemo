import os

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

## API Keys
# export/setx OPENAI_API_KEY="your_key"
# print("OPENAI_API_KEY: ", OPENAI_API_KEY)
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

## LLMs
# OpenAI cloud LLM
llm_openai_cloud = ChatOpenAI(
    model="gpt-5-nano",                                     # Pricing per 1M tokens: input=$0.05, cached_input=$0.005, output=$0.40
                                                            # Rough estimate: 1M tokens ≈ 750k words, i.e., 10 lakh tokens ≈ 7.5 lakh words
    api_key=openai_api_key, 
    temperature=0                                           # Controls randomness: 0 = deterministic/focused, higher values = more creative/random
    # max_tokens=80                                         TODO: Not working currently with gpt-5-nano in this setup
)

# Ollama local LLM
llm_ollama_local = OllamaLLM(
    model="tinyllama",
    temperature=0,
    # num_predict=500                                        # Ollama output token limit
)

# Ollama cloud LLM
llm_ollama_cloud = OllamaLLM(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        # Adds API key to request headers. Ex: Authorization: Bearer xxxxx
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    },
    temperature=0,
    num_predict=500                                          # Ollama output token limit
)

## methods
# f-string: You build the prompt yourself directly
def ask_openai(question: str) -> str:
    prompt = f"""
             Answer clearly and briefly.

             Question: {question}
             """
    response = llm_openai_cloud.invoke(prompt)
    return response.content

def ask_ollama_local(question: str) -> str:
    # Tiny models need simpler + stricter prompt
    prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"
    response = llm_ollama_local.invoke(prompt)
    return response

def ask_ollama_cloud(question: str) -> str:
    prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"
    response = llm_ollama_cloud.invoke(prompt)
    return response

# select mode -> [OpenAI, Ollama Local, Ollama Cloud]
def normalize_mode(mode: str) -> str:
    mode_map = {
        "1": "openai",
        "2": "ollama-local",
        "3": "ollama-cloud",
        "openai": "openai",
        "ollama-local": "ollama-local",
        "ollama-cloud": "ollama-cloud"
    }

    if mode not in mode_map:
        print("Invalid mode. Defaulting to ollama-cloud.\n")
        return "ollama-cloud"

    return mode_map[mode]

# check if API keys are present
def check_required_key(mode: str) -> bool:
    if mode == "openai" and not openai_api_key:
        print("Error: OPENAI_API_KEY is missing.")
        return False

    if mode == "ollama-cloud" and not ollama_api_key:
        print("Error: OLLAMA_API_KEY is missing.")
        return False

    return True

## MAIN PROGRAM
if __name__ == "__main__":
    print("Chatbot started! Type 'exit' to quit.\n")
    print("Available modes: ")
    print("1. openai")
    print("2. ollama-local")
    print("3. ollama-cloud")

    mode = input("Choose mode [1/2/3]: ").strip().lower()
    mode = normalize_mode(mode)

    if not check_required_key(mode):
        exit(1)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        if not user_input:
            continue

        try:
            if mode == "openai":
                answer = ask_openai(user_input)
            elif mode == "ollama-local":
                answer = ask_ollama_local(user_input)
            else:
                answer = ask_ollama_cloud(user_input)

            print("Bot:", answer)

        except Exception as e:
            print("Error:", str(e))


## Run:
# 1. Tab 1 (PowerShell): 
#    cd D:\dev\github\agentic-ai-langchaindemo
# 
#    # create venv (only first time)
#    # Creates isolated Python environment (prevents conflicts)
#    python -m venv venv311
# 
#    # allow scripts (only per session)
#    # Required because PowerShell blocks scripts by default
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# 
#    # activate venv
#    venv311\Scripts\activate
# 
#    # install deps (only first time)
#    pip install streamlit langchain langchain-ollama langchain-openai
# 
#    # start ollama server (if running locally)
#    ollama pull tinyllama
#    ollama list
#    ollama serve
#    
#    # If local Ollama is running, you’ll see JSON with models
#    curl http://localhost:11434/api/tags
#
# 
# 2. Tab 2 (PowerShell):
#    cd D:\dev\github\agentic-ai-langchaindemo\s4_langchain-in-action
#    python 1_openai_demo.py          