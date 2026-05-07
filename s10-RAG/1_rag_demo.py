import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama

## OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# =========================================================
# 1. EMBEDDING MODEL (using Ollama local)
#    Purpose: Understand/search text
# =========================================================
# Converts text into vectors (embeddings)
# Embeddings capture semantic meaning. Similar meaning → vectors close together
#
# Example:
# "I love Java" -> [0.123, -0.553, 0.991, ...] 
# These numbers are called Embeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# llm = ChatOllama(model="tinyllama")                        # Works well, but use Ollama cloud llm instead to avoid cpu stress


# =========================================================
# 2. LLM (GENERATION MODEL) (using Ollama cloud)
#    Purpose: Generate final answer
# =========================================================
# Used to generate final answer from retrieved context

# embeddings = OllamaEmbeddings(model="nomic-embed-text")    # Use Ollama local embeddings instead (because getting 401 unauthorized in this setup/account)
llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={                                                # Adds API key to request headers. Ex: Authorization: Bearer xxxxx
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)


# =========================================================
# 3. LOAD DOCUMENT
# =========================================================
document = TextLoader("product-data.txt", encoding="utf-8").load()


# =========================================================
# 4. SPLIT DOCUMENT INTO CHUNKS
# =========================================================
# Large documents are split into smaller chunks for better retrieval accuracy

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(document)


# =========================================================
# 5. CREATE VECTOR DATABASE FOR SEMANTIC SEARCH
# =========================================================
# Internally:
# chunk -> embedding vector -> store in Chroma
#
# Example:
# "Battery lasts 10 hours" -> [0.21, -0.88, ...]
# Normal DB: SELECT * WHERE id=5
# Vector DB: Find vectors semantically similar to: "battery backup"

vector_store = Chroma.from_documents(chunks, embeddings)    # Vector database. Stores embeddings for similarity search


# =========================================================
# 6. CREATE RETRIEVER
# =========================================================
# Retriever performs semantic search on vector DB later
#
# Question -> Convert to embedding -> Find relevant chunks from vector DB -> Return relevant chunks
# Ex: "What is battery life?"
# Retriever returns:
# [
#    "Battery lasts 10 hours",
#    "Supports fast charging"
# ]

retriever = vector_store.as_retriever()                     # Create a retriever object that can (do semantic) search this vector database later


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
qa_chain = create_stuff_documents_chain(llm, prompt_template)


# =========================================================
# 9. CREATE RAG PIPELINE
# =========================================================
# Full runtime flow:
#  - retriever: Question -> Find relevant chunks from vector DB -> Return documents
#  - qa_chain : Retrieved chunks + Question -> Build prompt -> Send to LLM -> LLM generates final answer 
#
# Equivalent mental model:
#   def rag_chain(question):
#       docs = retriever(question)
#       answer = qa_chain(question, docs)
#       return answer
rag_chain = create_retrieval_chain(retriever, qa_chain)         # search vector store and fetch relevant chunks for the given 'input'
                                                           

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
#       docs = retriever(question)
#       answer = qa_chain(question, docs)
#       return answer


# =========================================================
# 10. ASK QUESTIONS
# =========================================================
print("Chat with Document")
question = input("Your Question: ")

if question:
    # Hidden internal flow:
    #   rag_chain.invoke({"input": question})
    # 
    # Internally becomes roughly:
    #   retrieved_docs = retriever.invoke(question)
    #   answer = qa_chain.invoke({
    #       "context": retrieved_docs,
    #       "input": question
    #   })
    response = rag_chain.invoke(
        {
            "input": question
        }
    )
    print("\nAnswer:")
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