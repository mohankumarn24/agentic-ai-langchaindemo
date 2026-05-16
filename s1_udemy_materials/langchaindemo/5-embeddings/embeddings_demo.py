import os
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm=OpenAIEmbeddings(api_key=OPENAI_API_KEY)

text = input("Enter the text")
response = llm.embed_query(text)
print(response)