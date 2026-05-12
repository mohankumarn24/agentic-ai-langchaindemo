import os

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

## Embedding Models
# OpenAI cloud embedding api
# export/setx OPENAI_API_KEY="your_key"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# Ollama local embedding
# Uses an embedding model to convert text into numerical vectors
ollama_embeddings_local  = OllamaEmbeddings(model="nomic-embed-text")

# Ollama cloud embedding
# This didn't work initially, so upgraded langchain-ollama:
#     pip install -U langchain-ollama
#
# ollama_embeddings_cloud = OllamaEmbeddings(
#     model="nomic-embed-text",
#     base_url="https://ollama.com",
#     headers={
#         "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
#     }
# )
#
# 401 Unauthorized may mean the API key/account does not have permission for this endpoint/model.
#
# ollama_embeddings_cloud = OllamaEmbeddings(
#     model="nomic-embed-text",
#     base_url="https://ollama.com",
#     client_kwargs={
#         "headers": {
#             "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
#         }
#     }
# )

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
#     Generates vectors.

## Generate embedding
text = input("Enter text: ")
response = ollama_embeddings_local.embed_query(text)            # Converts one text string into one vector
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
# If running using Ollama Cloud: (401 issue - permission issues after upgrade too!!)
# pip install -U langchain-ollama
# pip show langchain-ollama
#
# cd D:\dev\github\agentic-ai-langchaindemo\s8_embeddings
# python 1_embeddings_demo.py
