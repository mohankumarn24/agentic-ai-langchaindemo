import os
import uuid
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

## OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# =========================================================
# 1. EMBEDDING MODEL (OLLAMA LOCAL)
# =========================================================
# Purpose:
# Convert text into embeddings/vectors.
#
# Embeddings capture semantic meaning.
# Similar meaning -> vectors become closer together
#
# Example:
# "I love Java"
# -> [0.123, -0.553, 0.991, ...]
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Optional local LLM
# llm = ChatOllama(model="tinyllama")

# =========================================================
# 2. LLM (OLLAMA CLOUD)
# =========================================================
# Purpose:
# Generate final natural-language answers.
#
# Embedding model:
#   Understand/search text
#
# LLM:
#   Generate final answer

# Below code throws getting 401 unauthorized in this setup/account
# embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="https://ollama.com", headers={"Authorization":f"Bearer {os.environ.get('OLLAMA_API_KEY')}"})   

llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

# =========================================================
# 3. LOAD DOCUMENT
# =========================================================
# Reads product-data.txt into LangChain Document objects

document = TextLoader(
    "product-data.txt", 
    encoding="utf-8"
).load()

# =========================================================
# 4. SPLIT DOCUMENT INTO CHUNKS
# =========================================================
# Large documents are split into smaller chunks for better retrieval accuracy and semantic search

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(document)

# =========================================================
# 5. CREATE VECTOR DATABASE (CHROMA)
# =========================================================
# Internally:
#
# chunk
#   -> embedding vector
#   -> stored in Chroma DB
#
# Example:
# "Battery lasts 10 hours"
# -> [0.21, -0.88, ...]

vector_store = Chroma.from_documents(chunks, embeddings)

# =========================================================
# 6. CREATE RETRIEVER
# =========================================================
# Retriever performs semantic search on vector DB later.
#
# Runtime retrieval flow:
#
# Question
# -> question embedding
# -> similarity search in Chroma
# -> relevant chunks returned

retriever = vector_store.as_retriever()

# =========================================================
# 7. HISTORY-AWARE RETRIEVER PROMPT
# =========================================================
# Purpose:
# Rewrite follow-up questions into standalone questions.
#
# Example:
#
# User:
# "Tell me about XYZ smartphone"
#
# Follow-up:
# "What about battery?"
#
# Internally rewritten into:
# "What is the battery capacity of XYZ smartphone?"
#
# MessagesPlaceholder("chat_history")
# dynamically injects previous conversation messages.

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

# =========================================================
# 8. CREATE HISTORY-AWARE RETRIEVER
# =========================================================
# Full internal flow:
#
# Current Question
# +
# Chat History
# -> LLM rewrites standalone question
# -> Retriever searches vector DB
# -> Relevant chunks returned

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# =========================================================
# 9. QA PROMPT (FINAL ANSWERING PROMPT)
# =========================================================
# Purpose:
# Generate final answer using retrieved chunks.
#
# Retrieved chunks get inserted into:
# {context}
#
# Chat history gets inserted into:
# {chat_history}
#
# Current user question gets inserted into:
# {input}

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

# =========================================================
# 10. CREATE QA CHAIN
# =========================================================
# "Stuff" means:
# Retrieved chunks are stuffed directly into prompt context.
#
# Equivalent mental model:
#
# def qa_chain(question, docs):
#     prompt = build_prompt(question, docs)
#     return llm(prompt)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)                     # Stuff means - retrieved chunks are stuffed into prompt context, then sent to LLM

# =========================================================
# 11. CREATE FULL RAG PIPELINE
# =========================================================
# Full runtime flow: 
# RAG chain = history aware retriever + QA chain
#
# User Question
# -> History-aware retriever rewrites question
# -> Retriever searches vector DB
# -> Relevant chunks returned
# -> QA chain builds prompt
# -> LLM generates final answer

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
)

# =========================================================
# 12. CHAT HISTORY STORAGE (STREAMLIT)
# =========================================================
# Stores conversation messages in Streamlit session

history_for_chain = StreamlitChatMessageHistory()

# =========================================================
# 13. WRAP RAG CHAIN WITH MEMORY SUPPORT
# =========================================================
# RunnableWithMessageHistory automatically:
#
# 1. Loads previous messages
# 2. Passes chat history into rag_chain
# 3. Stores new conversation messages
#
# Without RunnableWithMessageHistory,
# you would manually manage chat history yourself.

chain_with_history = RunnableWithMessageHistory(
    rag_chain,                                                              # Which chain to wrap
    lambda session_id : history_for_chain,                                  # Where chat messages are stored
    input_messages_key="input",                                             # User question field name
    history_messages_key="chat_history",                                    # History variable name inside prompts ie., previous messages  
                                                                            # Variable name used by MessagesPlaceholder("chat_history")               
    output_messages_key="answer",                                           # AI response field name
)

## Without RunnableWithMessageHistory, YOU would need to do:
#   chat_history = []
#   question = input()
#
#   response = rag_chain.invoke({
#       "input": question,
#       "chat_history": chat_history
#   })
# 
#   chat_history.append(question)
#   chat_history.append(response["answer"])    
#
# RunnableWithMessageHistory automates this annoying memory handling

# =========================================================
# 14. STREAMLIT UI
# =========================================================

st.write("Mini Customer Support AI")
question = st.text_input("Your Question")

# =========================================================
# 15. SESSION MANAGEMENT
# =========================================================
# Each browser session gets unique ID.
#
# This allows separate memory per session.

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

# =========================================================
# 16. EXECUTE CONVERSATIONAL RAG
# =========================================================

if question:

    # 1. Before (simple RAG): 
    # Only current question was sent into rag_chain
    # 
    #   rag_chain.invoke({
    #       "input": question
    #   })

    # 2. Current (conversational RAG): 
    # "Run rag_chain with automatic memory handling"
    # 
    # RunnableWithMessageHistory automatically:
    #   1. Loads old chat messages
    #   2. Passes chat history into rag_chain
    #   3. Stores new conversation messages
    #
    # Hidden internal flow:
    #   rag_chain.invoke({
    #       "input": question,
    #       "chat_history": previous_messages
    #   }) 
    # 
    # 3. chain_with_history.invoke(...) internally:
    #   -> loads previous chat history
    #   -> rewrites follow-up question (if needed)
    #   -> retrieves relevant docs
    #   -> injects context + chat history
    #   -> calls LLM
    #   -> stores new conversation messages
    #   -> returns answer

    # 4. Different capabilities of current RAG SYSTEM
    #   - NORMAL RAG RETRIEVAL
    #   - CONVERSATIONAL MEMORY
    #   - HISTORY-AWARE RETRIEVER
    #   - SEMANTIC RETRIEVAL
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

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s10-RAG
# python -m streamlit run 2_historyaware_rag_demo.py
# http://localhost:8501/   

# Ask Questions:
#   1. TEST NORMAL RAG RETRIEVAL
#      What are the features of XYZ smartphone?
#   2. TEST HISTORY-AWARE RETRIEVER
#      Tell me about XYZ smartphone
#   3. TEST HISTORY-AWARE RETRIEVER
#      What about battery?
#   4. TEST SEMANTIC RETRIEVAL
#      Can I send back electronics after purchase?


# ------------------------------------------------------------
# 1. TEST NORMAL RAG RETRIEVAL
# ------------------------------------------------------------

# User Question:
# What are the features of XYZ smartphone?

# Internal flow:
# 1. User question is converted into embeddings/vectors
# 2. Chroma vector DB performs semantic similarity search
# 3. Retriever finds relevant XYZ smartphone chunks from product-data.txt
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

# Internal flow:
# 1. Retriever searches vector DB for XYZ smartphone chunks
# 2. Relevant product information is retrieved
# 3. RunnableWithMessageHistory stores conversation
# 4. LLM generates answer using retrieved context


# ------------------------------------------------------------
# 3. TEST HISTORY-AWARE RETRIEVER
# ------------------------------------------------------------

# Follow-up Question:
# What about battery?

# Internal flow:
# 1. Previous chat history is loaded
# 2. History-aware retriever understands:
#       "battery" refers to XYZ smartphone
#
# 3. Internally rewritten into something like:
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

# Internal flow:
# 1. Exact sentence is NOT present in document
#
# 2. Embeddings understand semantic similarity:
#
#       "send back electronics"
#               ≈
#       "return policy"
#
# 3. Vector similarity search retrieves return-policy chunk
# 4. Retrieved context is injected into prompt
# 5. LLM generates natural-language response



## FINAL ARCHITECTURE
#
# User Question + Chat History
#        ↓
# History-Aware Retriever
#        ↓
# Standalone Question Rewrite
#        ↓
# Retriever Searches Vector DB
#        ↓
# Relevant Chunks
#        ↓
# QA Prompt + LLM
#        ↓
# Final Answer
#        ↓
# Store Updated Chat History
