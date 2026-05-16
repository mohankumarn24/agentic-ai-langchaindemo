import os
from langchain_openai import OpenAIEmbeddings
import numpy as np

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=OpenAIEmbeddings(api_key=OPENAI_API_KEY)

text1 = input("Enter the text1")
text2 = input("Enter the text2")
response1 = llm.embed_query(text1)
response2 = llm.embed_query(text2)

similarity_score = np.dot(response1,response2)

print(similarity_score*100,'%')