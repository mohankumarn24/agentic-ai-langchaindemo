import os
from langchain_community.chat_models import ChatOllama

llm=ChatOllama(model="mistral")

question = input("Enter the question")
response = llm.invoke(question)
print(response.content)