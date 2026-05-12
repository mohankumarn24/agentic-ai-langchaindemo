import os

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

## LLM
# OpenAI cloud api
# export/setx OPENAI_API_KEY="your_key"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# print("OPENAI_API_KEY: ", OPENAI_API_KEY)
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

## methods
# f-string: You build the prompt yourself directly.
def ask_openai(question: str) -> str:
    # OpenAI can handle raw or slightly guided prompts
    prompt = f"Answer clearly:\n{question}"
    response = openai_llm_cloud.invoke(prompt)
    return response.content


def ask_ollama(question: str) -> str:
    # Tiny models need simpler + stricter prompt
    prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"
    response = ollama_llm_local.invoke(prompt)
    return response


def ask_ollama_cloud(question: str) -> str:
    prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"
    response = ollama_llm_cloud.invoke(prompt)
    return response

## MAIN PROGRAM
if __name__ == "__main__":
    print("Chatbot started! Type 'exit' to quit.\n")

    mode = input("Choose mode (openai / ollama / ollama cloud): ").strip().lower()

    if mode not in ["openai", "ollama", "ollama cloud"]:
        print("Invalid mode. Defaulting to ollama cloud.\n")
        mode = "ollama cloud"

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break

        if not user_input:
            continue

        try:
            if mode == "openai":
                answer = ask_openai(user_input)
            elif mode == "ollama":
                answer = ask_ollama(user_input)
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
#    # start ollama server
#    ollama serve
# 
# 2. Tab 2 (PowerShell):
#    cd D:\dev\github\agentic-ai-langchaindemo
#    venv311\Scripts\activate
# 
#    cd D:\dev\github\agentic-ai-langchaindemo\s4_langchain-in-action
#    python 1_openai_demo.py          