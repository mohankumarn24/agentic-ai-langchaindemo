import os
import streamlit as st
import uuid

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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

## Prompt template: creates the final message structure sent to the LLM
# It has 3 parts - System message, Previous chat history, Current human question
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a Scrum Coach. Answer questions related to Scrum clearly."),            # Defines AI behavior/personality/context
    MessagesPlaceholder(variable_name="chat_history"),                                          # "POSITION where old messages are inserted"
                                                                                                # Previous chat history gets inserted here
                                                                                                # Ex:
                                                                                                # [
                                                                                                #     HumanMessage(content="Explain Scrum"),
                                                                                                #     AIMessage(content="Scrum is an Agile framework..."),
                                                                                                #
                                                                                                #     HumanMessage(content="Summarize this"),
                                                                                                #     AIMessage(content="Scrum is a lightweight Agile framework...")
                                                                                                # ]
    ("human","{input}")                                                                         # Current user question
])

## LCEL chain
chain = prompt_template | ollama_llm_cloud | StrOutputParser()

## Temporary in-memory conversation storage.
#  Stores previous user + AI messages temporarily.
#  Each session_id gets its own chat memory.
#
# Initially empty:
#       {}
#
# Later:
#    {
#       "session-1": InMemoryChatMessageHistory([HumanMessage(content="What is Scrum?"), AIMessage(content="Scrum is an Agile framework...")]),
#       "session-2": InMemoryChatMessageHistory([HumanMessage(content="Who is Scrum Mster?"), AIMessage(content="Scrum Master is...")]),
#    }

store = {}
def get_history(session_id):
    if session_id not in store:
        # New session_id -> create new chat memory
        store[session_id] = InMemoryChatMessageHistory()            # LangChain memory object that stores messages in normal Python memory
                                                                    # When the program stops, memory is gone

    # Existing session_id -> return existing chat memory
    return store[session_id]

## Adds conversational memory support to the chain.
#  RunnableWithMessageHistory is a wrapper that automatically:
#     1. Gets old messages using get_history(session_id)
#     2. Inserts them into MessagesPlaceholder(variable_name="chat_history")
#     3. Sends current input to the model
#     4. Stores current user question and AI response back into history

## Preparation happens here
# Create a wrapper around chain.
# Tell wrapper:
#    - use get_history to fetch memory
#    - current input key is "input"
#    - history placeholder key is "chat_history"

chain_with_history = RunnableWithMessageHistory(
    chain,                                          # Original LCEL chain

    get_history,                                    # "WHERE to get/store history"
                                                    # session_id: "WHICH conversation memory to use"
                                                    # This tells RunnableWithMessageHistory where to get the history from.
                                                    # RunnableWithMessageHistory reads the session_id from config and internally calls: get_history(session_id)
                                                    # session_id helps separate conversations:
                                                    #     Same session_id      -> same chat history
                                                    #     Different session_id -> different chat history
                                                    # get_history(session_id):
                                                    #     Returns existing history for session_id if present.
                                                    #     Otherwise creates a new InMemoryChatMessageHistory.

    input_messages_key="input",                     # "WHERE to read current user input"
                                                    # The current user question is inside the dictionary key called "input".
                                                    # Example:
                                                    #     chain_with_history.invoke(
                                                    #         {"input": question},
                                                    #         config={...}
                                                    #     )

    history_messages_key="chat_history"             # "WHERE to put history inside the prompt"
                                                    # Put previous messages into the prompt variable called "chat_history"          
                                                    # Must match MessagesPlaceholder(variable_name="chat_history").
                                                    # Old conversation messages are injected into: chat_history.
)
                                                    ## The model receives below info when second question is asked:
                                                    # System:
                                                    # You are a Scrum Coach...
                                                    # 
                                                    # Chat history:
                                                    # Human: What is Scrum?
                                                    # AI: Scrum is a lightweight Agile framework...
                                                    # 
                                                    # Human:
                                                    # Summarize this in one sentence

## Generate
print("Scrum Guide")
session_id = str(uuid.uuid4())
while True:
    # Debugging: 
    # Prints stored conversation memory: 
    #   (previous HumanMessage + AIMessage objects)
    # 
    # print(get_history(session_id).messages)

    question = input("\nEnter question related to Scrum: ")
    if question.lower() == "exit":
        print("\nExiting chatbot. Good Bye!!")
        break

    if question:
        # Execution happens here
        response = chain_with_history.invoke(
            {
                "input": question                   # Current user input
            },
            {
                "configurable":                     # Runtime configuration
                {
                    "session_id": session_id        # Same session_id -> same chat history
                                                    # Different session_id -> fresh/separate conversation.
                }
            }
        )
        print("\nAI Response:")
        print(response)
        print("-" * 50)


## Flow:
# 1. RunnableWithMessageHistory wraps the original LCEL chain.
# 2. When chain_with_history.invoke(...) is called, execution starts.
# 3. It reads session_id from config["configurable"]["session_id"].
# 4. It calls get_history(session_id) to get the correct chat history.
# 5. It reads the current user question from the input_messages_key, i.e., "input".
# 6. It inserts old messages into MessagesPlaceholder(variable_name="chat_history").
# 7. It inserts the current question into ("human", "{input}").
# 8. It executes the original chain:
#        prompt_template -> ollama_llm_cloud -> StrOutputParser
# 9. After the AI response is generated, it stores the new HumanMessage and AIMessage.
# 10. It returns the final response.



## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s7_maintaining-chathistory
# python 2_chathistory_demo.py
# http://localhost:8501/          

# After enabling RunnableWithMessageHistory:
#
# Round 1:
#     Human: what is scrum
#     AI   : Scrum is a lightweight Agile framework...
#
# Round 2:
#     Human: summarize in one sentence
#     AI   : Summarizes the previous Scrum explanation
#
# Why it works:
#     RunnableWithMessageHistory stores previous HumanMessage + AIMessage objects.
#     MessagesPlaceholder(variable_name="chat_history") injects those old messages
#     into the next LLM call.
#
# So the second question can refer to "this" or "summarize" because the model
# receives the previous conversation context.



## Stored chat history:
#   [
#       HumanMessage(content="what is scrum"),
#       AIMessage(content="long Scrum explanation..."),
#   
#       HumanMessage(content="summarize in one sentence"),
#       AIMessage(content="short Scrum summary...")
#   ]