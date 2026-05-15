import os

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

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

# Numerical vectors
#   These vectors capture the meaning of the text.
#   They are used for semantic search / retrieval.


## Embeddings
# Embeddings convert text into numerical vectors.
# Embeddings help AI understand how similar or related different words/sentences are.
# They represent text based on semantic meaning and relationships with other text.
#
# Similar meaning   -> vectors are close.
# Different meaning -> vectors are far apart.
#
# LLM:
#     Generates text.
#
# Embedding model:
#     Generates numerical vectors
#       - These vectors capture the meaning of the text
#       - They are used for semantic search / retrieval.

## Generate embedding
text = input("Enter text: ")
response = openai_embeddings_cloud.embed_query(text)            # Using OpenAI embeddings
                                                                # Converts one text string into one vector
print(response)                                                 # List of floating-point numbers representing semantic meaning.
print("Vector size:", len(response))


## Note
# gpt-oss:20b       -> chat/generation model
# gpt-4o-mini       -> chat/generation model
# text-embedding-*  -> OpenAI embedding/vector model
# nomic-embed-text  -> Ollama embedding/vector model


## Run:
# If running using Ollama Local:
# ollama pull nomic-embed-text
# 
# If running using Ollama Cloud
# pip install -U langchain-ollama
# pip show langchain-ollama
#
# cd D:\dev\github\agentic-ai-langchaindemo\s8_embeddings
# python 1_embeddings_demo.py
