import os

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

## 1. OpenAI Cloud API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

## 2. Ollama local
llm_ollamaLocal = OllamaEmbeddings(model="nomic-embed-text")

## 3. Ollama cloud API key
# This didn't work. So upgraded langchain-ollama -> "pip install -U langchain-ollama"
# llm_ollamaCloud = OllamaEmbeddings(
#     model="nomic-embed-text",
#     base_url="https://ollama.com",
#     headers={
#         "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
#     }
# )

# Some 401 issue. Seems like my account doesn't has permissions
llm_ollamaCloud = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
        }
    }
)

# What are Embeddings?
# Embeddings help AI understand how similar or related different words and sentences are.
# They represent text as numerical vectors based on semantic meaning and relationships with other words/texts.
# Similar meanings    -> similar vectors
# Different meanings  -> distant vectors

text = input("Enter text: ")
response = llm_ollamaLocal.embed_query(text)
print(response)




## Note
# gpt-oss:20b       -> chat/generation model
# nomic-embed-text  -> embedding/vector model


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

