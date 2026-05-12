import os
import streamlit as st
import uuid

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

## LLM
# OpenAI cloud api
# export/setx OPENAI_API_KEY="your_key"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_llm_cloud = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

# Ollama cloud api
ollama_llm_local = ChatOllama(model="tinyllama")
ollama_llm_cloud = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={                                                # Adds API key to request headers. Ex: Authorization: Bearer xxxxx
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

## Prompt template
prompt_template = ChatPromptTemplate.from_messages([    
    ("system", "You are an Agile Coach. Answer questions related to Agile process clearly."),       # Defines AI behavior/personality/context

    MessagesPlaceholder(variable_name="chat_history"),                                              # Previous chat history gets inserted here
                                                                                                    # Without this, AI receives only the latest question.
                                                                                                    # MessagesPlaceholder is used for chat history injection

    ("human", "{input}")                                                                            # Current user question
])

## LCEL chain
chain = (
    prompt_template
    | ollama_llm_cloud
    | StrOutputParser()
)

## Temporary in-memory conversation storage.
# StreamlitChatMessageHistory stores messages inside:
#     st.session_state["chat_messages"]
#
# Default key:
#     "langchain_messages"
#
# Custom key:
#     StreamlitChatMessageHistory(key="chat_messages")
history_for_chain = StreamlitChatMessageHistory()           # LangChain memory object, but specially made for Streamlit
                                                            # Stores chat messages in st.session_state.
                                                            # Good for Streamlit apps because Streamlit reruns the script after user interactions.

## Adds conversational memory support to the chain.
#  RunnableWithMessageHistory is a wrapper that automatically:
#     1. Gets old messages using the history function.
#     2. Inserts them into MessagesPlaceholder(variable_name="chat_history").
#     3. Sends current input to the model.
#     4. Stores current user question and AI response back into history.

chain_with_history = RunnableWithMessageHistory(
    chain,                                      # Original LCEL chain
    lambda session_id : history_for_chain,      # Returns the Streamlit chat memory object
                                                # Same as:
                                                #     def get_history(session_id):
                                                #         return history_for_chain
                                                    
    input_messages_key="input",                 # Current user input field name
    history_messages_key="chat_history"         # Matches MessagesPlaceholder variable
)

## Streamlit UI
st.title("Agile Guide")
user_input = st.text_input("Enter question:")

## Streamlit session id
#  Streamlit reruns the entire script for every interaction.
#  So session_id should be stored in st.session_state.
#  Otherwise a new session_id may be generated on every rerun.
#
#  Ex: 
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

## Generate
if st.button("Ask"):
    if user_input:
        response = chain_with_history.invoke(
            {
                "input": user_input
            },
            {
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        st.subheader("Answer")
        st.markdown(response)
    else:
        st.warning("Please enter a question")

## DEBUG MEMORY
st.subheader("Chat History")

# Prints stored HumanMessage + AIMessage objects.
st.write(history_for_chain.messages)

## Run:
# cd D:\dev\github\agentic-ai-langchaindemo
# pip install langchain-community
# pip show langchain-community

# cd D:\dev\github\agentic-ai-langchaindemo\s7_maintaining-chathistory
# python -m streamlit run 3_streamlit_chathistory_demo.py
# http://localhost:8501/   

## Questions:
# What is Scrum
# Who is Scrum master
# Can you summarize in two sentences