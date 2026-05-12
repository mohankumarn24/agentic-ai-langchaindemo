import os
import numpy as np

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

# print(os.environ.get("OLLAMA_API_KEY"))

text1 = input("Enter first text     : ")
text2 = input("Enter second text    : ")
response1 = ollama_embeddings_local.embed_query(text1)
response2 = ollama_embeddings_local.embed_query(text2)

## Dot product compares vectors directly, but larger vectors can give misleadingly high scores
# similarity_score = np.dot(response1, response2)
# print(similarity_score*100, '%')

## Cosine similarity checks how similar the meanings of two texts are, regardless of vector size/length
#  
#   Note: Cosine similarity is usually interpreted as a score from -1 to 1, not a real percentage
#   For embeddings, common meaning:
#       0.80 - 1.00  -> very similar
#       0.50 - 0.80  -> somewhat related
#       0.00 - 0.50  -> weakly related
#       < 0          -> opposite/unrelated direction
similarity_score = np.dot(response1, response2) / (np.linalg.norm(response1) * np.linalg.norm(response2))
print("Similarity Score      : ", similarity_score)
print("Similarity Percentage : ", round(similarity_score * 100, 2), "%")


## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s8_embeddings
# python 2_similarity_finder.py

## Output:
# Note: Ideally 'Similarity Score' must be 1. 
#       But because computers store decimal numbers approximately, so sometimes you get a value extremely close to 1, but not exactly 1
# 
# Enter first text        : hello
# Enter second text       : hello
# Similarity Score        : 0.9999999999999998          
# Similarity Percentage   : 100.0 %

# Enter first text        : hello
# Enter second text       : hi
# Similarity Score        : 0.8871811574470746
# Similarity Percentage   : 88.72 %                     # 88.72% is just a friendly display, not an exact “88.72% same meaning”
