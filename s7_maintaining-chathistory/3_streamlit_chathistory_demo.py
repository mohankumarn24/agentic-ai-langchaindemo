import os
import streamlit as st
import uuid

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

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
# Stores messages inside:
# st.session_state["chat_messages"]
history_for_chain = StreamlitChatMessageHistory()

# history_for_chain = StreamlitChatMessageHistory(
#    key="chat_messages"
# )

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

# STREAMLIT UI
st.title("Agile Guide")

user_input  = st.text_input("Enter question:")

## STREAMLIT SESSION ID
# Streamlit re-runs entire script for every interaction
# So session_id must be stored in session_state. Otherwise chatbot memory resets every rerun
# Ex: 
#       st.session_state = 
#                   {
#                       "session_id": "550e8400-e29b-41d4-a716-446655440000" ,
#                       "key1"      : "value1",
#                       "key2"      : "value2",
#                       "key3"      : "value3"
#                   }
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

if user_input :
    response = chain_with_history.invoke(
        {
            "input":user_input 
        },
        {
            "configurable": 
                {
                    "session_id": session_id
                }
        }
    )
    st.write(response)

## DEBUG MEMORY
# st.subheader("Chat History")

# Prints stored HumanMessage + AIMessage objects
# st.write(history_for_chain.messages)

## Run:
# cd D:\dev\github\agentic-ai-langchaindemo
# pip install langchain-community
# pip show langchain-community

# cd D:\dev\github\agentic-ai-langchaindemo\s7_maintaining-chathistory
# python -m streamlit run 3_streamlit_chathistory_demo.py
# http://localhost:8501/   