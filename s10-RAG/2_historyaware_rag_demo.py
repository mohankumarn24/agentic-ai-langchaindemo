import os
import streamlit as st
import uuid

from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)

## OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

## 1.a. Ollama local
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# llm = ChatOllama(model="tinyllama")                        # Works well, but use Ollama cloud llm instead to avoid cpu stress

## 1.b. Ollama Cloud
# embeddings = OllamaEmbeddings(model="nomic-embed-text")    # Use Ollama local embeddings instead (because getting 401 unauthorized in this setup/account)
llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

# Load & split documents
document = TextLoader("product-data.txt", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(document)

# Vector store & retriever
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# 1. Prompt for history-aware retriever
# (NO {context} HERE)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that reformulates follow-up
            questions into standalone questions.

            Use the chat history and the latest user input to create
            a self-contained question.

            Do NOT answer the question, only rewrite it."""
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human", 
            "{input}"
        ),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# 2. Prompt for answering with context
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant for answering questions.
            Use the provided context to respond.
            If the answer isn't clear from the context, say you don't know.
            Limit your response to three concise sentences.

            Context:
            {context}
            """
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human", 
            "{input}"
        ),
    ]
)   
qa_chain = create_stuff_documents_chain(llm, qa_prompt)                     # Stuff means - retrieved chunks are stuffed into prompt context, then sent to LLM

# RAG chain = history aware retriever + QA chain
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
)

# Chat history (Streamlit)
history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id : history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",                                           # final answer key from create_retrieval_chain
)

# Streamlit UI
st.write("Mini Customer Support AI")
question = st.text_input("Your Question")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

if question:
    # retrieves docs -> injects context -> calls LLM -> returns answer
    response = chain_with_history.invoke(
        {
            "input": question
        },
        {
            "configurable": 
                {
                    "session_id": session_id
                }
        }        
    )
    st.write(response['answer'])

## RAG
# Question -> Chat History -> History-Aware Retriever -> Standalone Query Rewrite -> Retriever -> Relevant Chunks -> LLM Answer -> Store Memory
# 
# Conversational.
# Understands follow-up questions.
# Remembers earlier chat.
#
# Like talking to ChatGPT continuously.
# It remembers context.


## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s10-RAG
# python -m streamlit run 2_historyaware_rag_demo.py
# http://localhost:8501/   






# ------------------------------------------------------------
# 1. TEST NORMAL RAG RETRIEVAL
# ------------------------------------------------------------

# User Question:
# What are the features of XYZ smartphone?

# What happens internally:
# 1. User question is converted into embeddings/vectors
# 2. Chroma vector DB searches for semantically similar chunks
# 3. Retriever finds matching XYZ smartphone information
#    from product-data.txt
# 4. Retrieved context is injected into prompt
# 5. LLM generates final human-readable answer

# Example Answer:
# The XYZ smartphone features a 6.5-inch display,
# 128 GB storage, 6 GB RAM, 48 MP camera,
# and 4000 mAh battery.


# ------------------------------------------------------------
# 2. TEST CONVERSATIONAL MEMORY
# ------------------------------------------------------------

# User Question:
# Tell me about XYZ smartphone

# What happens internally:
# 1. Retriever searches vector DB for XYZ smartphone chunks
# 2. Relevant product information is retrieved
# 3. Chat history is stored in memory
# 4. LLM answers using retrieved context


# ------------------------------------------------------------
# 3. TEST HISTORY-AWARE RETRIEVER
# ------------------------------------------------------------

# Follow-up Question:
# What about battery?

# What happens internally:
# 1. Chat history is checked
# 2. History-aware retriever understands:
#       "battery" refers to XYZ smartphone
#
# 3. Internally question becomes something like:
#       "What is the battery capacity of XYZ smartphone?"
#
# 4. Retriever searches vector DB again
# 5. Battery-related chunk is retrieved
# 6. LLM generates final response


# ------------------------------------------------------------
# 4. TEST SEMANTIC RETRIEVAL
# ------------------------------------------------------------

# User Question:
# Can I send back electronics after purchase?

# What happens internally:
# 1. Exact sentence is NOT present in document
#
# 2. Embeddings understand semantic meaning:
#
#       "send back electronics"
#               ≈
#       "return policy"
#
# 3. Vector similarity search retrieves return-policy chunk
# 4. Retrieved context is sent to LLM
# 5. LLM generates natural language response