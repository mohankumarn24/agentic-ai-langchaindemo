import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings, ChatOllama

## OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

## 1.a. Ollama local
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# llm = ChatOllama(model="tinyllama")                        # Works well, but use Ollama cloud llm instead to avoid cpu stress

## 1.b. Ollama Cloud
# embeddings = OllamaEmbeddings(model="nomic-embed-text")    # Use Ollama local embeddings instead (because getting 401 unauthorized in this setup/account)
llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization":
            f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

print(os.getcwd())
document = TextLoader("product-data.txt", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(document)
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant for answering questions.
            
            Use the provided context to respond. If the answer 
            isn't clear, acknowledge that you don't know. 
            
            Limit your response to three concise sentences.
            
            Context:
            {context}
            """
        ),
        (
            "human", 
            "{input}"
        )
    ]
)

# 
qa_chain = create_stuff_documents_chain(llm, prompt_template)   # Stuff means - retrieved chunks are stuffed into prompt context, then sent to LLM
rag_chain = create_retrieval_chain(retriever, qa_chain)         # Retriever + QA chain

print("Chat with Document")
question = input("Your Question: ")

if question:
    # retrieves docs -> injects context -> calls LLM -> returns answer
    response = rag_chain.invoke(
        {
            "input": question
        }
    )
    print(response['answer'])

## Simple RAG
# Stateless. Every question independent.
# 
# Question -> Retriever -> Relevant Chunks -> LLM Answer
# No memory. No conversation awareness. Every question is treated independently.
# 
# Ex: You ask 'Tell me about XYZ smartphone' and then ask 'What about battery?'
#     retriever only sees 'What about battery?'. It does NOT know 'battery of WHAT?'. So retrieval may become weak/wrong
#
# Like asking Google search repeatedly.
# Each search independent.

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s10-RAG
# python 1_rag_demo.py

## Output 1
# Chat with Document
# Your Question: What is the status of my order #98765?
# I’m sorry, but I don’t have the specifics for order #98765. You can view its status by logging into your account and navigating to the “My Orders” section. 
# If you have a tracking number from your confirmation email, you can also use it to check the latest shipment updates.

## Output 2
# Chat with Document
# Your Question: How do I apply for a job at your company?
# I’m sorry, but I don’t have information in this context about applying for a job at our company.  
# If you’re interested in opportunities, please visit our website’s Careers page for current openings and application instructions.  
# Alternatively, feel free to email HR at jobs@example.com for further assistance.