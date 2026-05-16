import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

document = PyPDFLoader("academic_research_data.pdf").load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=200)
chunks=text_splitter.split_documents(document)
vector_store=Chroma.from_documents(chunks,embeddings)
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","""You are an assistant for answering questions.
    Use the provided context to respond.If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
    
    """),
    ("human", "{input}")
]
)

qa_chain = create_stuff_documents_chain(llm,prompt_template)
rag_chain = create_retrieval_chain(retriever,qa_chain)

print("Chat with Document")
question=input("Your Question")

if question:
    response = rag_chain.invoke({"input":question})
    print(response['answer'])