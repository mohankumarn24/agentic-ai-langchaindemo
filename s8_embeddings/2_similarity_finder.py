import os
import numpy as np

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

## 1. OpenAI Cloud API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

## 2. Ollama local
llm_ollamaLocal = OllamaEmbeddings(model="nomic-embed-text")

## 3. Ollama cloud API key
# Still some 401 issue. Seems like my account doesn't has permissions
llm_ollamaCloud = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
        }
    }
)

# print(os.environ.get("OLLAMA_API_KEY"))

text1 = input("Enter first text     : ")
text2 = input("Enter second text    : ")
response1 = llm_ollamaLocal.embed_query(text1)
response2 = llm_ollamaLocal.embed_query(text2)

## Dot product compares vectors directly, but larger vectors can give misleadingly high scores
# similarity_score = np.dot(response1, response2)
# print(similarity_score*100,'%')

## Cosine similarity checks how similar the meaning of two texts are, regardless of vector size/length
similarity_score = np.dot(response1, response2) / (np.linalg.norm(response1) * np.linalg.norm(response2))
print("Similarity Score      : ", similarity_score)
print("Similarity Percentage : ", round(similarity_score * 100, 2), "%")


## Run:
# If running using Ollama Local:
# ollama pull nomic-embed-text
# 
# If running using Ollama Cloud: (401 issue - permission issues after upgrade too!!)
# pip install -U langchain-ollama
# pip show langchain-ollama
#
# cd D:\dev\github\agentic-ai-langchaindemo\s8_embeddings
# python 2_similarity_finder.py

## Output
# Enter first text        : hello
# Enter second text       : hello
# Similarity Score        : 1.0
# Similarity Percentage   : 100.0 %

# Enter first text        : hello
# Enter second text       : hi
# Similarity Score        : 0.8871811574470746
# Similarity Percentage   : 88.72 %