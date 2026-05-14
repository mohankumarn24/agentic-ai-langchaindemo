import os

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from  langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="deepseek-r1:1.5b")

document = TextLoader("product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=200)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an assistant for answering questions.
        Use the provided context to respond.If the answer 
        isn't clear, acknowledge that you don't know. 
        Limit your response to three concise sentences.
        {context}
        """
    ),
    (
        "human", 
        "{input}"
    )
])

qa_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)

print("Chat with Deepseek Document")
question=input("Your Question: ")

if question:
    response = rag_chain.invoke({"input": question})
    print(response['answer'])


## Run
# TODO: May stress CPU
# cd D:\dev\github\agentic-ai-langchaindemo\s14_use-deepseek-models
# python 2_deepseek_rag_demo.py   
#
# Q: What is the status of my order #98765?
# Q: How do I apply for a job at your company?