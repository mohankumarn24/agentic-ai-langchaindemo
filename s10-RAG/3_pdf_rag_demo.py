import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


## API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

# =========================================================
# 1. EMBEDDING MODEL (OLLAMA LOCAL)
# =========================================================
# Purpose:
# Convert text into embeddings/vectors.
#
# Embeddings capture semantic meaning.
# Similar meaning -> vectors become closer together
#
# Example:
# "I love Java"
# -> [0.123, -0.553, 0.991, ...]

# OpenAI cloud embedding model
openai_embeddings_cloud = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=openai_api_key
)

# Ollama local embedding model
ollama_embeddings_local = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================================================
# 2. LLM (OLLAMA CLOUD)
# =========================================================
# Purpose:
# Generate final natural-language answers.
#
# Embedding model:
#   Understand/search text
#
# LLM:
#   Generate final answer

# OpenAI cloud LLM
llm_openai_cloud = ChatOpenAI(
    model="gpt-5-nano", 
    api_key=openai_api_key, 
    temperature=0                                           # Controls randomness: 0 = deterministic/focused, higher values = more creative/random
)

# Ollama cloud LLM
llm_ollama_cloud = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        # Adds API key to request headers. Example: Authorization: Bearer xxxxx
        "Authorization": f"Bearer {ollama_api_key}"
    },
    temperature=0
)


# =========================================================
# 3. LOAD PDF DOCUMENT
# =========================================================
# PyPDFLoader extracts text from PDF pages.
#
# Each PDF page becomes a LangChain Document object.
#
# Example:
#
# [
#   Document(
#       page_content="Page 1 text...",
#       metadata={"page": 0}
#   ),
#
#   Document(
#       page_content="Page 2 text...",
#       metadata={"page": 1}
#   )
# ]

document = PyPDFLoader(
    "3_academic_research_data.pdf"
).load()


# =========================================================
# 4. SPLIT PDF INTO CHUNKS
# =========================================================
# Large PDF text is split into smaller chunks.
#
# Why?
# - Better retrieval accuracy
# - Better semantic matching
# - Avoid huge prompts
# - LLM context window limitations
#
# chunk_overlap preserves continuity between chunks.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)
chunks = text_splitter.split_documents(document)


# =========================================================
# 5. CREATE VECTOR DATABASE (CHROMA)
# =========================================================
# Internally:
#
# chunk
#   -> embedding vector
#   -> stored in Chroma DB
#
# Example:
#
# "Neural networks improve accuracy"
# -> [0.21, -0.88, ...]
#
# Traditional DB:
# SELECT * WHERE id=5
#
# Vector DB:
# Find semantically similar vectors.
#
# Example semantic search:
# "deep learning"
# may match:
# "neural networks"

# Vector store. Stores embeddings for similarity search
persist_directory = "chroma_db_openai"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    # load existing saved Chroma DB from disk
    print("Loading existing Chroma DB...")
    vector_store = Chroma(
        embedding_function=openai_embeddings_cloud,
        persist_directory=persist_directory
    )
else:
    # create embeddings from documents and save them to disk
    print("Creating new Chroma DB...")
    vector_store = Chroma.from_documents(
        documents=chunks,                                    # List of documents to add to the VectorStore
        embedding=openai_embeddings_cloud,
        persist_directory=persist_directory
    )

# =========================================================
# 6. CREATE RETRIEVER
# =========================================================
# Retriever performs semantic search on vector DB later.
#
# Runtime retrieval flow:
#
# Question
# -> question embedding
# -> similarity search in Chroma
# -> relevant chunks returned
#
# Example:
#
# Question:
# "What improves model accuracy?"
#
# Retriever may return:
# [
#   "Neural networks improved accuracy by 25%",
#   "Deep learning achieved better results"
# ]

retriever = vector_store.as_retriever()


# =========================================================
# 7. CREATE PROMPT TEMPLATE
# =========================================================
# Purpose:
# Define how LLM should answer.
#
# Retrieved chunks get inserted into:
# {context}
#
# Current user question gets inserted into:
# {input}

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for answering questions.
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
    ]
)


# =========================================================
# 8. CREATE QA CHAIN
# =========================================================
# "Stuff" means:
# Retrieved chunks are stuffed directly into prompt context.
#
# Equivalent mental model:
#
#   def qa_chain(question, docs):
#       prompt = build_prompt(question, docs)
#       return llm(prompt)
#
# Example final prompt:
#
# Context:
# Neural networks improved accuracy by 25%
#
# Question:
# What improved model accuracy?

qa_chain = create_stuff_documents_chain(llm_openai_cloud, prompt_template)


# =========================================================
# 9. CREATE FULL RAG PIPELINE
# =========================================================
# Full runtime flow:
#
# User Question
# -> Retriever searches vector DB
# -> Relevant PDF chunks returned
# -> QA chain builds prompt
# -> LLM generates final answer
#
# Equivalent mental model:
#
#   def rag_chain(question):
#       docs = retriever(question)
#       answer = qa_chain(question, docs)
#       return answer
rag_chain = create_retrieval_chain(retriever, qa_chain)


# =========================================================
# 10. ASK QUESTIONS
# =========================================================

print("Chat with Document")
question = input("Your Question: ")

if question:
    # Before create_retrieval_chain():
    # You would manually do:
    #
    #   docs = retriever.invoke(question)
    #   response = qa_chain.invoke({
    #       "context": docs,
    #       "input": question
    #   })

    # Hidden internal flow:
    #
    # chain.invoke(...)
    #   -> converts question into embedding
    #   -> retriever searches vector DB
    #   -> relevant PDF chunks returned
    #   -> chunks injected into prompt
    #   -> LLM generates final answer
    #   -> returns response    
    response = rag_chain.invoke(
        {
            "input": question
        }
    )
    print("\nAnswer:")
    print(response["answer"])

## Run
# cd D:\dev\github\agentic-ai-langchaindemo
# pip install pypdf

# cd D:\dev\github\agentic-ai-langchaindemo\s10-RAG
# python 3_pdf_rag_demo.py

# Ask questions:
# 1. What are the key findings from recent studies on the impact of climate change on coral reefs?
# 2. How has ocean warming affected fish populations according to recent research?
# 3. What adaptations have marine species developed in response to climate change?
# 4. What are the predicted consequences of ocean acidification on marine food webs?
# 5. What strategies have been suggested for marine conservation in response to climate change?


# =========================================================
# FINAL ARCHITECTURE
# =========================================================
#
# PDF Document
#        ↓
# Extract PDF text
#        ↓
# Split into chunks
#        ↓
# Convert chunks into embeddings
#        ↓
# Store embeddings in Chroma vector DB
#        ↓
# User asks question
#        ↓
# Retriever performs semantic search
#        ↓
# Relevant PDF chunks returned
#        ↓
# Chunks inserted into prompt
#        ↓
# LLM generates final answer
#
#
# =========================================================
# TEST EXAMPLES
# =========================================================
#
# ---------------------------------------------------------
# 1. TEST BASIC PDF RAG
# ---------------------------------------------------------
#
# User Question:
# "What is the main conclusion of the research?"
#
# Internal flow:
# 1. Question converted into embeddings
# 2. Retriever searches PDF vector DB
# 3. Relevant research chunks retrieved
# 4. Chunks injected into prompt
# 5. LLM generates answer
#
#
# ---------------------------------------------------------
# 2. TEST SEMANTIC SEARCH
# ---------------------------------------------------------
#
# User Question:
# "How did the model improve performance?"
#
# Even if exact wording does not exist:
#
# "improve performance"
#            ≈
# "increase accuracy"
#
# semantic similarity search may still retrieve:
#
# "The neural network increased accuracy by 25%"
#
#
# ---------------------------------------------------------
# 3. TEST UNKNOWN ANSWERS
# ---------------------------------------------------------
#
# User Question:
# "What is the CEO salary mentioned in the paper?"
#
# If answer is absent from retrieved context:
# LLM should say:
# "I don't know"
#
# because prompt instructs model not to hallucinate.
#