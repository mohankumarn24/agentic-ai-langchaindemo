import os

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

## API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

## Embedding Models
# Uses an embedding model to convert text into numerical vectors
# OpenAI cloud embedding model
openai_embeddings_cloud = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key
)

# Ollama local embedding model
ollama_embeddings_local = OllamaEmbeddings(
    model="nomic-embed-text"
)

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

###################################################################################################
## 1c Option 1. In-memory Chroma vector database
# Store text chunks + embeddings in Chroma vector database.
# 
# This creates Chroma only in memory:
# Program starts -> Embeddings are created -> Search works -> Program ends -> Vector DB is gone
#
#
#   inmemory_chroma_db = Chroma.from_documents(
#       documents=chunks,
#       embedding=openai_embeddings_cloud
#   )
###################################################################################################

## 1c Option 2. Persistent Chroma vector database
# Chroma.from_documents does the same thing as above, but also saves the DB to disk.
#
# Adding persistent storage makes embeddings persist between runs.
# This creates a local folder named "chroma_db_openai" and stores the vector database there.
#
# Program starts -> Embeddings are created -> Saved inside chroma_db_openai folder -> Search works -> Program ends -> Vector DB remains saved

## See Note (IMPORTANT -> to reduce pricing!!)
persist_directory = "chroma_db_openai"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    # load existing saved Chroma DB from disk
    print("Loading existing Chroma DB...")
    persistent_chroma_db = Chroma(
        embedding_function=openai_embeddings_cloud,
        persist_directory=persist_directory
    )
else:
    # create embeddings from documents and save them to disk
    print("Creating new Chroma DB...")
    persistent_chroma_db = Chroma.from_documents(
        documents=chunks,                                    # List of documents to add to the VectorStore
        embedding=openai_embeddings_cloud,
        persist_directory=persist_directory
    )

# Currently using persistent DB
retriever = persistent_chroma_db.as_retriever()

# 2a. Enter user question
text = input("Enter the query: ")

## 2b. Semantic retrieval/search
# Internally:
#  - Converts user query into embedding/vector/numbers by making api call to embedding model
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







## Note
## OpenAI Embedding Flow with Persistent Chroma DB
#
# FIRST RUN / CREATE DB:
# 1. Load document from file.
# 2. Split document into chunks.
# 3. Call OpenAI embedding model for each document chunk.
# 4. Convert each chunk into an embedding/vector.
# 5. Store original chunk text + embedding/vector in persistent Chroma DB.
# 6. Save the Chroma DB inside "chroma_db_openai" folder.
#
# WHEN USER ASKS A QUESTION:
# 1. Call OpenAI embedding model for the user question.
# 2. Convert user question into an embedding/vector.
# 3. Compare question embedding with document embeddings already stored in Chroma DB.
# 4. Return the most semantically similar chunks.
#
# SECOND RUN / LOAD EXISTING DB:
# 1. Do not recreate embeddings for the same old document chunks.
# 2. Load already saved document embeddings from "chroma_db_openai".
# 3. Old document embeddings are reused from persistent DB.
#
# WHEN USER ASKS A QUESTION AGAIN:
# 1. Call OpenAI embedding model for the new user question.
# 2. Convert user question into an embedding/vector.
# 3. Compare question embedding with saved document embeddings in Chroma DB.
# 4. Return the most semantically similar chunks.
#
# Simple rule:
# - Document chunks are embedded only when creating/updating the DB.
# - User questions are embedded every time the user searches.



## When is persistent Chroma DB updated?
#
# Persistent DB is updated only when documents are written to it.
#
# DB is updated in these cases:
# 1. Creating DB:
#       Chroma.from_documents(...)
#    -> Embeds document chunks and stores them in DB.
#
# 2. Adding new documents:
#       persistent_chroma_db.add_documents(new_chunks)
#    -> Embeds new chunks and stores them in DB.
#
# 3. Deleting/updating records manually:
#       persistent_chroma_db.delete(ids=[...])
#    -> Removes records from DB.
#
# DB is NOT updated when user asks a question:
#       docs = retriever.invoke(user_question)
#    -> Only embeds the question and searches existing vectors.
#    -> Does not save the question.
#    -> Does not modify saved document embeddings.