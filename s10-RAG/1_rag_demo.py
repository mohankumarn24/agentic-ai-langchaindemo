import os

from typing import Tuple, Union
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama

## API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

# =========================================================
# 1. EMBEDDING MODEL (using Ollama local)
#    Purpose: Understand/search text
# =========================================================
# Embedding model converts text into vectors (embeddings)
# These vectors are used for semantic search
# Embeddings capture semantic meaning. Similar meaning → vectors close together

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
# 2. LLM (GENERATION MODEL) (using Ollama cloud)
#    Purpose: Generate final answer
# =========================================================
# LLM generates the final answer using retrieved context

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
document = TextLoader("product-data.txt", encoding="utf-8").load()


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
# 7. CREATE PROMPT TEMPLATE
# =========================================================
# Creates prompt template for LLM
# Retrieved chunks get inserted into {context}
# User question gets inserted into {input}

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant for answering questions.
            
            Use the provided context to respond. If the answer 
            isn't clear, acknowledge that you don't know. 
            
            Limit your response to three concise sentences.
            
            Context:
            {context}
            """
        ),
        (
            "human", 
            "{input}"
        )
    ]
)

# Suppose user asks: 
#   What is battery life?
# 
# Retriever finds: 
#   Battery lasts 10 hours.
# 
# Final prompt becomes:
#   You are an assistant for answering questions.
#   
#   Context: 
#   Battery lasts 10 hours
#   
#   Human:
#   What is battery life?
#
# Then sent to LLM.


# =========================================================
# 8. CREATE QA CHAIN
# =========================================================
# "Stuff" means: Retrieved chunks are stuffed directly into prompt context, then sent to LLM
#
# Stuff chain inserts all retrieved chunks into the prompt
# 
# Ask LLM using retrieved chunks + question to generate answer
#   Example final prompt:
#       Context:
#       Battery lasts 10 hours.
# 
#       Question:
#       What is battery life?
# 
#       LLM answers:
#       Battery lasts 10 hours.
#
# Equivalent mental model:
#   def qa_chain(question, docs):
#       prompt = build_prompt(question, docs)
#       return llm(prompt)

# helper function
def select_llm_provider() -> Tuple[str, Union[ChatOpenAI, ChatOllama]]:
    print("Choose LLM provider:")
    print("1. OpenAI Cloud")
    print("2. Ollama Cloud")

    choice = input("Enter choice [1/2]: ").strip()

    if choice == "1":
        print("Selected LLM: OpenAI Cloud")
        return "OpenAI Cloud", llm_openai_cloud

    if choice == "2":
        print("Selected LLM: Ollama Cloud")
        return "Ollama Cloud", llm_ollama_cloud

    print("Invalid choice. Defaulting to Ollama Cloud.")
    return "Ollama Cloud", llm_ollama_cloud

selected_provider, llm_selected = select_llm_provider()
qa_chain = create_stuff_documents_chain(llm_selected, prompt_template)


# =========================================================
# 9. CREATE RAG PIPELINE
#    RAG chain = retriever + QA chain
#    Question -> Retriever -> Relevant chunks -> LLM -> Answer
# =========================================================
# Full runtime flow:
#  - retriever: Question -> Find relevant chunks from vector DB -> Return documents
#  - qa_chain : Retrieved chunks + Question -> Build prompt -> Send to LLM -> LLM generates final answer 
#
# 1. retriever = vector_store.as_retriever()
#    Equivalent mental model:
#    def retriever(question):
#       return similar_chunks
# 
# 2. qa_chain = create_stuff_documents_chain(...)
#    Equivalent mental model:
#    def qa_chain(question, docs):
#       prompt = f"""
#                Context:
#                {docs}
# 
#                Question:
#                {question}
#                """
#        return llm(prompt)
#
# 3. rag_chain = create_retrieval_chain(retriever, qa_chain)
#    Equivalent mental model:
#    def rag_chain(question):
#       relevant_docs = retriever(question)
#       llm_answer = qa_chain(question, relevant_docs)
#       return llm_answer
rag_chain = create_retrieval_chain(retriever, qa_chain)         # search vector store and fetch relevant chunks for the given 'input'
                                                           

# =========================================================
# 10. ASK QUESTIONS
# =========================================================
print("Chat with Document")
question = input("Your Question: ")

if question:
    # ========================================================
    # WHAT rag_chain.invoke({"input": question}) DOES INTERNALLY
    # ========================================================
    #
    # rag_chain is an orchestrator.
    # It combines:
    #
    #   1. retriever
    #      - searches Chroma
    #      - returns relevant document chunks
    #
    #   2. qa_chain
    #      - takes retrieved chunks + user question
    #      - puts them into the prompt template
    #      - sends final prompt to LLM
    #      - returns final answer

    # Hidden internal flow:
    #   rag_chain.invoke({"input": question})
    # 
    # Internally becomes roughly:
    #   def rag_chain_invoke(inputs):
    #       question = inputs["input"]
    #   
    #       # STEP 1: Retrieve relevant chunks from vector DB
    #       retrieved_docs = retriever.invoke(question)
    #   
    #       # STEP 2: Send question + retrieved chunks to LLM chain
    #       answer = qa_chain.invoke({
    #           "context": retrieved_docs,
    #           "input": question
    #       })
    #   
    #       # STEP 3: Return complete response
    #       return {
    #           "input": question,
    #           "context": retrieved_docs,
    #           "answer": answer
    #       }
    
    response = rag_chain.invoke(
        {
            "input": question
        }
    )
    print("\nAnswer: ")
    print(response['answer'])



# =========================================================
# SIMPLE RAG ARCHITECTURE
# =========================================================
#
# Text File
#    ↓
# Split into chunks
#    ↓
# Convert chunks into embeddings
#    ↓
# Store embeddings in vector DB (Chroma)
#    ↓
# User asks question
#    ↓
# Retriever finds relevant chunks
#    ↓
# Chunks + Question sent to LLM
#    ↓
# LLM generates answer
#
#
# NOTE:
# This is stateless RAG.
# Every question is independent.
#
# Example:
#
# Q1: Tell me about iPhone 15
# Q2: What about battery?
#
# Retriever only sees:
# "What about battery?"
#
# It does NOT automatically know:
# "battery of iPhone 15"
#
# Conversational RAG solves this later.




## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s10-RAG
# python 1_rag_demo.py

## Output 1
# Chat with Document
# Your Question: What is the status of my order #98765?
# I’m sorry, but I don’t have the specifics for order #98765. You can view its status by logging into your account and navigating to the “My Orders” section. 
# If you have a tracking number from your confirmation email, you can also use it to check the latest shipment updates.

## Output 2
# Chat with Document
# Your Question: How do I apply for a job at your company?
# I’m sorry, but I don’t have information in this context about applying for a job at our company.  
# If you’re interested in opportunities, please visit our website’s Careers page for current openings and application instructions.  
# Alternatively, feel free to email HR at jobs@example.com for further assistance.