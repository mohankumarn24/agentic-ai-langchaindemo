import os
import uuid

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

## 1. OpenAI Cloud API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

## 2. Ollama local
llm_ollamaLocal = OllamaLLM(model="tinyllama")

## 3. Ollama cloud API key
llm_ollamaCloud = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

## PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # Defines AI behavior/personality
            "You are an Agile Coach. "
            "Answer questions related to Agile process clearly."
        ),

        # Previous chat history gets inserted here
        # Without this, AI will forget earlier conversation
        # MessagesPlaceholder is used for chat history injection
        MessagesPlaceholder(variable_name="chat_history"),

        (
            "human",
            # Current user question
            "{input}"
        )
    ]
)

# LCEL CHAIN
# chain = prompt_template | llm_ollamaLocal | StrOutputParser()
chain = (
    prompt_template
    | llm_ollamaCloud
    | StrOutputParser()
)

# Temporary in-memory conversation storage
history_for_chain = ChatMessageHistory()

# Adds conversational memory support to chain
# Wrapper that automatically:
#  - Retrieves old messages
#  - Injects memory into prompt
#  - Stores latest conversation
chain_with_history = RunnableWithMessageHistory(
    # Original LCEL chain
    chain,
    # Returns memory object
    lambda session_id : history_for_chain,
    # Current user input field name
    input_messages_key="input",
    # Matches MessagesPlaceholder variable
    history_messages_key="chat_history"
)

#   def get_history(session_id):
#        return history_for_chain
# 
# is same as:
#   lambda session_id: history_for_chain

print("Agile Guide")
session_id = str(uuid.uuid4())

while True:
    question = input("Enter question: ")
    if question:
        response = chain_with_history.invoke(
            {
                "input":question 
            },
            {
                "configurable": 
                    {
                        "session_id": session_id
                    }
            }
        )
    print(response)

## Run:
# cd D:\dev\github\agentic-ai-langchaindemo
# pip install langchain-community
# pip show langchain-community

# cd D:\dev\github\agentic-ai-langchaindemo\s7_maintaining-chathistory
# python 4_chathistory_demo.py  