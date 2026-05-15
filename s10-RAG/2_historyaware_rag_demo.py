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

## API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

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

# OpenAI cloud embedding model
openai_embeddings_cloud = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key
)

# Ollama local embedding model
ollama_embeddings_local = OllamaEmbeddings(
    model="nomic-embed-text"
)


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

# OpenAI cloud LLM
llm_openai_cloud = ChatOpenAI(
    model="gpt-5-nano", 
    api_key=openai_api_key, 
    temperature=0                                           # Controls randomness: 0 = deterministic/focused, higher values = more creative/random
)

# Ollama cloud LLM
llm_ollama_cloud = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        # Adds API key to request headers. Example: Authorization: Bearer xxxxx
        "Authorization": f"Bearer {ollama_api_key}"
    },
    temperature=0
)

# =========================================================
# 3. LOAD DOCUMENT
# =========================================================
document = TextLoader(
    "product-data.txt", 
    encoding="utf-8"
).load()


# =========================================================
# 4. SPLIT DOCUMENT INTO CHUNKS
# =========================================================
# Large documents are split into smaller chunks
# chunk_overlap keeps some context between chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(document)


# =========================================================
# 5. CREATE VECTOR DATABASE FOR SEMANTIC SEARCH
# =========================================================
# Chroma stores embeddings and allows semantic search
# 
# Internally:
# chunk -> embedding vector -> store in Chroma
#
# Example:
# "Battery lasts 10 hours" -> [0.21, -0.88, ...]
# Normal DB: SELECT * WHERE id=5
# Vector DB: Find vectors semantically similar to: "battery backup"

# Vector store. Stores embeddings for similarity search
persist_directory = "chroma_db_openai"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    # load existing saved Chroma DB from disk
    print("Loading existing Chroma DB...")
    vector_store = Chroma(
        embedding_function=openai_embeddings_cloud,
        persist_directory=persist_directory
    )
else:
    # create embeddings from documents and save them to disk
    print("Creating new Chroma DB...")
    vector_store = Chroma.from_documents(
        documents=chunks,                                    # List of documents to add to the VectorStore
        embedding=openai_embeddings_cloud,
        persist_directory=persist_directory
    )

# =========================================================
# 6. CREATE RETRIEVER
# =========================================================
# Retriever performs semantic search on vector DB later
#
# Retriever searches the vector DB and returns relevant chunks
# k=3 means return top 3 most relevant chunks.
# 
#   retriever = vector_store.as_retriever(
#       search_kwargs={"k": 3}
#   )
#
# Question -> Convert to embedding -> Find relevant chunks from vector DB -> Return relevant chunks
# Ex: "What is battery life?"
# Retriever returns:
# [
#    "Battery lasts 10 hours",
#    "Supports fast charging"
# ]

# Currently using persistent DB
retriever = vector_store.as_retriever()             # Create a retriever object that can (do semantic) search this vector database later 

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

            Do NOT answer the question, only rewrite it.
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
# 8. CREATE HISTORY-AWARE RETRIEVER
# =========================================================
# Converts follow-up questions into standalone questions before retrieval.
#
# Flow:
#   input + chat_history
#       -> LLM rewrites input using contextualize_q_prompt
#       -> standalone question
#       -> retriever searches vector DB
#       -> relevant chunks returned
#
# If there is no chat_history:
#   input may be sent directly to retriever.
#
# Important:
#   This retrieves documents only.
#   It does not generate the final answer.

history_aware_retriever = create_history_aware_retriever(
    llm_openai_cloud,
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
#
# qa_chain is the answer-generation part of RAG.
#
# "Stuff" means:
#   Retrieved chunks/documents are directly stuffed into
#   the prompt variable, usually {context}.
#
# It does NOT retrieve documents.
# It only takes already-retrieved documents and asks the LLM
# to generate the final answer.
#
# Equivalent mental model:
#
#   def qa_chain(input, context, chat_history=None):
#       prompt = build_prompt(
#           input=input,              # current user question
#           context=context,          # retrieved document chunks
#           chat_history=chat_history # previous messages, if prompt uses it
#       )
#
#       answer = llm.invoke(prompt)
#       return answer
#
# Runtime:
#   retrieved docs + user question + optional chat history
#       -> prompt
#       -> LLM
#       -> final answer
#
# Important:
#   qa_chain does not search Chroma/vector DB.
#   Retrieval already happened before this chain runs

qa_chain = create_stuff_documents_chain(llm_openai_cloud, qa_prompt)                     # Stuff means - retrieved chunks are stuffed into prompt context, then sent to LLM

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


# 1. history_aware_retriever
# ------------------------------------------------------------
# history_aware_retriever = create_history_aware_retriever(...)
#
# Equivalent mental model:
#
#   def history_aware_retriever(input, chat_history):
#
#       # If there is previous conversation, use LLM to rewrite
#       # the current follow-up question into a standalone question.
#       if chat_history exists:
#           standalone_question = rewrite_follow_up_question(input, chat_history)
#       else:
#           standalone_question = input
#
#       # Send standalone question to normal vector retriever.
#       #
#       # retriever.invoke(...) internally:
#       #   1. Converts standalone_question into embedding/vector
#       #   2. Compares it with stored chunk vectors in Chroma
#       #   3. Finds semantically similar chunks
#       #   4. Returns those chunks as documents
#       relevant_docs = retriever.invoke(standalone_question)
#       return relevant_docs
#
# Job:
#   - Look at current question + previous chat history
#   - Convert follow-up question into standalone question
#   - Search vector store using standalone question
#   - Return relevant chunks/documents
#
# Example:
#   chat_history:
#       Human: What is Samsung S24 storage?
#       AI: Samsung S24 has 256GB storage.
#
#   current input:
#       What about iPhone?
#
#   rewritten standalone question:
#       What is the storage of iPhone?
#
#   retriever searches vector DB using:
#       "What is the storage of iPhone?"
#
#
# 2. qa_chain = create_stuff_documents_chain(...)
# ------------------------------------------------------------
# Equivalent mental model:
#
#   def qa_chain(input, context, chat_history):
#       prompt = f"""
#       Chat History:
#       {chat_history}
#
#       Context:
#       {context}
#
#       Question:
#       {input}
#       """
#       return llm(prompt)
#
# Job:
#   - Take retrieved chunks as context
#   - Take current question as input
#   - Optionally take chat history if prompt includes it
#   - Build final prompt
#   - Send final prompt to LLM
#   - Return final answer
#
#
# 3. rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
# ------------------------------------------------------------
# Equivalent mental model:
#
#   def rag_chain(inputs):
#       input = inputs["input"]
#       chat_history = inputs.get("chat_history", [])
#
#       relevant_docs = history_aware_retriever(
#           input=input,
#           chat_history=chat_history
#       )
#
#       llm_answer = qa_chain.invoke({
#           "input": input,
#           "chat_history": chat_history,
#           "context": relevant_docs
#       })
#
#       return {
#           "input": input,
#           "chat_history": chat_history,
#           "context": relevant_docs,
#           "answer": llm_answer
#       }
#
# Job:
#   - Use chat history to understand follow-up questions
#   - Retrieve relevant chunks from vector store
#   - Send chunks + question to LLM
#   - Return answer
#
# Important:
#   create_retrieval_chain(...) only builds the pipeline.
#   It does not run retrieval yet.
#
#   Retrieval happens when you call:
#       rag_chain.invoke(...)

rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

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
    rag_chain,                                            # Chain being wrapped with memory

    lambda session_id : history_for_chain,                # Function that returns chat history for a session_id
                                                          # NOTE:
                                                          # This ignores session_id and always returns the same history object.
                                                          # Fine for learning/demo.
                                                          # For real apps, use a dictionary/database keyed by session_id.

    input_messages_key="input",                           # The key where the current user question is found
                                                          # Example:
                                                          #   chain_with_history.invoke({"input": question}, config)
                                                          #
                                                          # RunnableWithMessageHistory reads this "input" value
                                                          # and stores it as the latest human message.

    history_messages_key="chat_history",                  # The key used to inject previous messages into the wrapped chain
                                                          #
                                                          # Example hidden input to rag_chain:
                                                          #
                                                          #   {
                                                          #       "input": question,
                                                          #       "chat_history": previous_messages
                                                          #   }
                                                          #
                                                          # This must match:
                                                          #   MessagesPlaceholder("chat_history")

    output_messages_key="answer",                         # The key where the final AI answer is found in rag_chain output
                                                          #
                                                          # Example rag_chain output:
                                                          #
                                                          #   {
                                                          #       "input": question,
                                                          #       "context": retrieved_docs,
                                                          #       "answer": "Final answer from LLM"
                                                          #   }
                                                          #
                                                          # RunnableWithMessageHistory reads response["answer"]
                                                          # and stores it as the latest AI message.
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
question = st.text_input("Ask a question about the document")

ask_button = st.button("Ask")

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

    # 1. Before (simple RAG): 
    # Only the current question was sent into rag_chain.
    # Simple RAG had no memory of previous questions
    #
    #   response = rag_chain.invoke({
    #       "input": question
    #   })

    # 2. Now: Conversational RAG
    # We are using chain_with_history.
    #
    # chain_with_history is usually created using:
    #
    #   RunnableWithMessageHistory(...)
    #
    # Its job is memory handling.
    #
    # It automatically:
    #   1. Uses session_id to find old chat messages
    #   2. Loads previous chat messages
    #   3. Passes those messages into the wrapped chain
    #   4. Runs the wrapped chain
    #   5. Stores the new user question and AI answer

    # 3. Hidden internal flow of chain_with_history.invoke(...)
    # --------------------------------------------------------
    # This:
    #
    #   chain_with_history.invoke(
    #       {"input": question},
    #       {"configurable": {"session_id": session_id}}
    #   )
    #
    # roughly becomes:
    #
    #   previous_messages = get_chat_history(session_id)
    #   response = rag_chain.invoke({
    #       "input": question,
    #       "chat_history": previous_messages
    #   })
    #   save_to_chat_history(session_id, question, response["answer"])
    #   return response

    # 4. chain_with_history.invoke(...) internally:
    #   -> loads previous chat history
    #   -> rewrites follow-up question (if needed)
    #   -> retrieves relevant docs
    #   -> injects context + chat history
    #   -> calls LLM
    #   -> stores new conversation messages
    #   -> returns answer

    # 5. If you are using history-aware retriever
    # --------------------------------------------------------
    # The history-aware retriever handles follow-up questions.
    #
    # It does:
    #
    #   chat_history + current question
    #       ↓
    #   rewrite current question into standalone question
    #       ↓
    #   send standalone question to vector retriever
    #       ↓
    #   retrieve relevant documents
    #
    # Example:
    #
    #   Chat history:
    #       User: What is Samsung S24 storage?
    #       AI: Samsung S24 has 256GB storage.
    #
    #   Current question:
    #       What about iPhone?
    #
    #   Rewritten question:
    #       What is the storage of iPhone?


    # 6. Full current system capabilities
    # --------------------------------------------------------
    # Your current RAG system can have:
    #
    #   - Semantic retrieval
    #       Finds relevant chunks from vector DB.
    #
    #   - Normal RAG answering
    #       Sends retrieved context + question to LLM.
    #
    #   - Conversational memory
    #       Remembers previous user/AI messages by session_id.
    #
    #   - History-aware retrieval
    #       Rewrites follow-up questions using chat history.


    # Important:
    #   RunnableWithMessageHistory  = memory loader/saver
    #   History-aware retriever     = follow-up question rewriter
    #   Retriever                   = vector DB searcher
    #   QA chain                    = answer generator

if ask_button and question:
    with st.spinner(f"Thinking..."):
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

    st.write(response["answer"])

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
