import os

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

## Embedding Models
# OpenAI cloud embedding api
# export/setx OPENAI_API_KEY="your_key"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

## Ollama local embedding model
# Embedding models convert text into vectors/numbers.
# These vectors capture the meaning of the text.
# They are used for semantic search / retrieval.
ollama_embeddings_local = OllamaEmbeddings(model="nomic-embed-text")

## 1a. Read the text file
document = TextLoader("job_listings.txt").load()

## 1b. Split document into smaller chunks
# Smaller chunks improve retrieval quality
# chunk_overlap preserves a little context between chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=10
)
chunks = text_splitter.split_documents(document)

## 1c Option 1. In-memory Chroma vector database
# Store text chunks + embeddings in Chroma vector database.
#
# Chroma.from_documents does this internally:
#  - Takes each text chunk
#  - Uses "nomic-embed-text" to convert chunk text into vectors/numbers
#  - Stores original chunk text + vector in Chroma
#  - Enables semantic similarity search
#
# This creates Chroma only in memory:
# Program starts -> Embeddings are created -> Search works -> Program ends -> Vector DB is gone
inmemory_chroma_db = Chroma.from_documents(
    documents=chunks,
    embedding=ollama_embeddings_local
)

## 1c Option 2. Persistent Chroma vector database
# Chroma.from_documents does the same thing as above, but also saves the DB to disk.
#
# Adding persistent storage makes embeddings persist between runs.
# This creates a local folder named "chroma_db" and stores the vector database there.
#
# Program starts -> Embeddings are created -> Saved inside chroma_db folder -> Search works -> Program ends -> Vector DB remains saved
persistent_chroma_db = Chroma.from_documents(
    documents=chunks,
    embedding=ollama_embeddings_local,
    persist_directory="chroma_db"
)

## 1d. Create retriever/search engine
# Retriever searches the selected vector database and returns relevant chunks.
#
# To limit number of returned chunks:
# retriever = inmemory_chroma_db.as_retriever(
#     search_kwargs={"k": 2}
# )
#
# Currently using temporary in-memory DB:

retriever = inmemory_chroma_db.as_retriever()

# To use persistent DB instead:
# retriever = persistent_chroma_db.as_retriever()

# 2a. Enter user question
text = input("Enter the query: ")

## 2b. Semantic retrieval/search
# Internally:
#  - Converts user query into embedding/vector/numbers
#  - Compares query vector with stored chunk vectors
#  - Retrieves most similar chunks
docs = retriever.invoke(text)

## 2c. Print the matching job listings/chunks
for doc in docs:
    print(doc.page_content)


## Flow
# 1. Read job_listings.txt
# 2. Split it into small chunks
# 3. Convert chunks into embeddings/numbers
# 4. Store chunks + embeddings in Chroma
# 5. Take user query
# 6. Convert query into embedding/numbers
# 7. Compare query embedding with stored chunk embeddings
# 8. Return most similar chunks
# 9. Print those chunks


## Run
# cd D:\dev\github\agentic-ai-langchaindemo
# pip install langchain_chroma
#
# cd D:\dev\github\agentic-ai-langchaindemo\s9_vector-stores
# python job_search_helper.py



## Note: 
#  There is NO LLM in this code currently
#  This code does -> embedding + vector storage + semantic retrieval. NOT answer generation using LLM


## Logic
#  Imagine your file has job listings: 
#  
#  job_listings.txt
#     Java Backend Developer required with Spring Boot, REST APIs, Microservices.
#     Python Data Engineer required with Spark, Airflow, SQL.
#     Frontend Developer required with React, JavaScript, CSS.
# 
# Now you type:
#     spring boot job
# 
# Your program searches the file and prints
#     Java Backend Developer required with Spring Boot, REST APIs, Microservices.