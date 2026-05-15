import os
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -----------------------------
# OpenAI setup
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)


# -----------------------------
# Load & split documents
# -----------------------------
document = TextLoader("product-data.txt").load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(document)

# -----------------------------
# Vector store & retriever
# -----------------------------
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()


# -----------------------------
# 1️⃣ Prompt for history-aware retriever
#    (NO {context} HERE)
# -----------------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that reformulates follow-up
questions into standalone questions.

Use the chat history and the latest user input to create
a self-contained question.

Do NOT answer the question, only rewrite it."""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)


# -----------------------------
# 2️⃣ Prompt for answering with context
#    (HERE we use {context})
# -----------------------------
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for answering questions.
Use the provided context to respond.
If the answer isn't clear from the context, say you don't know.
Limit your response to three concise sentences.

Context:
{context}
"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)


# -----------------------------
# RAG chain = history aware retriever + QA chain
# -----------------------------
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
)


# -----------------------------
# Chat history (Streamlit)
# -----------------------------
history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",   # final answer key from create_retrieval_chain
)


# -----------------------------
# Streamlit UI
# -----------------------------
st.write("Chat with Document")
question = st.text_input("Your Question")

if question:
    response = chain_with_history.invoke(
        {"input": question},
        {"configurable": {"session_id": "abc123"}},
    )
    st.write(response["answer"])