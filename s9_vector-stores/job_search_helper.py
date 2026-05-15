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
# Chroma.from_documents does this internally:
#  - Takes each text chunk
#  - Uses "text-embedding-3-small" to convert chunk text into vectors/numbers
#  - Stores original chunk text + vector in Chroma
#  - Enables semantic similarity search
#
# This creates Chroma only in memory:
# Program starts -> Embeddings are created -> Search works -> Program ends -> Vector DB is gone

## In-memory Chroma flow
# Program starts
# -> Read document
# -> Split document into chunks
# -> Create embeddings for chunks
# -> Store chunks + embeddings only in memory/RAM
# -> Search works while program is running
# -> Program ends
# -> In-memory vector DB is deleted/lost
# -> Next run creates embeddings again

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
#
# Important:
# Chroma.from_documents(...) creates embeddings for the provided documents.
# To reuse an existing persistent DB without recreating document embeddings,
# load it later using Chroma(persist_directory=..., embedding_function=...).

## Persistent Chroma flow
# Program starts
# -> Read document
# -> Split document into chunks
# -> Create embeddings for chunks
# -> Store chunks + embeddings in Chroma
# -> Save vector DB to disk inside persist_directory
# -> Search works while program is running
# -> Program ends
# -> Vector DB remains saved on disk
# -> Later, the saved DB can be loaded instead of recreating document embeddings

## See Note (IMPORTANT -> to reduce pricing!!)
persist_directory = "chroma_db_openai"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    # load existing saved Chroma DB from disk
    print("Loading existing Chroma DB...")
    persistent_chroma_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=openai_embeddings_cloud
    )
else:
    # create embeddings from documents and save them to disk
    print("Creating new Chroma DB...")
    persistent_chroma_db = Chroma.from_documents(
        documents=chunks,
        embedding=openai_embeddings_cloud,
        persist_directory=persist_directory
    )

## 1d. Create retriever/search engine
# Retriever searches the selected vector database and returns relevant chunks.
#
# To limit number of returned chunks:
# retriever = inmemory_chroma_db.as_retriever(
#     search_kwargs={"k": 2}
# )
#

# Currently using persistent DB
retriever = persistent_chroma_db.as_retriever()

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







## Note 1
# Chroma has 2 common modes
# 
# 1. Create mode: Chroma.from_documents(...)
#       persistent_chroma_db = Chroma.from_documents(
#           documents=chunks,
#           embedding=openai_embeddings_cloud,
#           persist_directory="chroma_db_openai"
#       )
# 
# This means:
#       Create embeddings now and save them to disk
# 
# So every time this line runs, it does:
#   job_listings.txt
#       -> chunks
#       -> call OpenAI embeddings API ($$$$)
#       -> save vectors into chroma_db_openai
# 
# So it saves the DB, but it also recreates document embeddings when this code runs
#
# 2. Load mode: Chroma(...)
#       persistent_chroma_db = Chroma(
#           persist_directory="chroma_db_openai",
#           embedding_function=openai_embeddings_cloud
#       )
# 
# This means:
#   Load already saved embeddings from disk
# 
# It does not:
#   - read your documents again
#   - call OpenAI embeddings API again for old document chunks ($$$$)
#   - split chunks again
#   - recreate embeddings for old chunks
# 
# It only opens the already saved Chroma DB folder.
#
# Note:
# Even in load mode, the user query is still embedded at search time.
# So OpenAI embedding API is still called for each search query.







## Note 2
# When is OpenAI embedding model called?
#
# OpenAI embedding model is called whenever text must be converted into vectors.
#
# It is called in these cases:
#
# 1. Creating vector DB from documents:
#       Chroma.from_documents(
#           documents=chunks,
#           embedding=openai_embeddings_cloud,
#           persist_directory="chroma_db_openai"
#       )
#    -> Calls OpenAI embeddings API for all document chunks.
#
# 2. Adding new documents later:
#       persistent_chroma_db.add_documents(new_chunks)
#    -> Calls OpenAI embeddings API for the new document chunks.
#
# 3. Searching/querying:
#       docs = retriever.invoke("spring boot job")
#    -> Calls OpenAI embeddings API for the user query.
#
# It is NOT called for old document chunks when loading an existing DB:
#       persistent_chroma_db = Chroma(
#           persist_directory="chroma_db_openai",
#           embedding_function=openai_embeddings_cloud
#       )
#    -> Loads saved vectors from disk.
#    -> Does not recreate embeddings for old chunks.
#
# Simple rule:
#   Text -> Vector = embedding model/API is called.
#   Saved vector -> Loaded from disk = embedding model/API is not called.