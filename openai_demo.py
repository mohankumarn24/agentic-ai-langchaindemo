# ============================================================
# LLM Wrapper: OpenAI + Ollama
# ============================================================

# If you are using OpenAI, then make sure you have set environment variable:
# export/setx OPENAI_API_KEY="your_key"

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# print("OPENAI_API_KEY: ", OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

# Initialize
openai_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ollama_llm = OllamaLLM(model="tinyllama")


# ---------- OPENAI ----------
def ask_openai(question: str) -> str:
    # OpenAI can handle raw or slightly guided prompts
    prompt = f"Answer clearly:\n{question}"
    response = openai_llm.invoke(prompt)
    return response.content


# ---------- OLLAMA ----------
def ask_ollama(question: str) -> str:
    # Tiny models need simpler + stricter prompt
    prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"
    response = ollama_llm.invoke(prompt)
    return response


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    print("🤖 Chatbot started! Type 'exit' to quit.\n")

    mode = input("Choose mode (openai / ollama): ").strip().lower()

    if mode not in ["openai", "ollama"]:
        print("Invalid mode. Defaulting to ollama.\n")
        mode = "ollama"

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
            else:
                answer = ask_ollama(user_input)

            print("Bot:", answer)

        except Exception as e:
            print("Error:", str(e))

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

   python openai_demo.py

"""            