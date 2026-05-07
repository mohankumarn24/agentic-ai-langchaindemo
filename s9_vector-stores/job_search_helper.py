import os
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

## 1. OpenAI Cloud API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

## 2. Ollama local embedding model
# Embedding models convert text → vectors
# Used for:
#  - semantic search
#  - vector databases
#  - RAG retrieval
embedding = OllamaEmbeddings(model="nomic-embed-text")
# There is NO LLM in this code currently
# This code does -> embedding + vector storage + semantic retrieval. NOT answer generation using LLM

# Load text file
document = TextLoader("job_listings.txt").load()

# Split document into smaller chunks
#  - Vector databases work better with smaller text pieces
#  - Improves retrieval/search accuracy
#  - chunk_overlap preserves some context between chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=10
)
chunks = text_splitter.split_documents(document)

# Store embeddings in Chroma vector database
#  - Converts text chunks → embeddings/vectors
#  - Stores vectors for semantic similarity search
db = Chroma.from_documents(chunks, embedding)
# Add persistent storage: So embeddings persist between runs.
# Otherwise: embeddings recreated every execution
#   db = Chroma.from_documents(
#       chunks,
#       embedding,
#       persist_directory="chroma_db"
#   )

# Create retriever
#  - Searches vector database
#  - Returns most semantically relevant chunks
retriever = db.as_retriever()
# Limit retrieved docs:
#   retriever = db.as_retriever(
#       search_kwargs={"k": 2}
#   )

text = input("Enter the query: ")

# Semantic retrieval
# Internally:
#  - User query → embedding/vector
#  - Compares query vector with stored vectors
#  - Retrieves most similar chunks
docs = retriever.invoke(text)

# Print retrieved chunks
for doc in docs:
    print(doc.page_content)


## Run
# cd D:\dev\github\agentic-ai-langchaindemo
# pip install langchain_chroma
#
# cd D:\dev\github\agentic-ai-langchaindemo\s9_vector-stores
# python job_search_helper.py
