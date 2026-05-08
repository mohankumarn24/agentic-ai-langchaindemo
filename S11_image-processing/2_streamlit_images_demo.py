import os
import base64
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# =========================================================
# 1. LLM
# =========================================================

## OpenAI cloud vision model
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#
# llm = ChatOpenAI(
#     model="gpt-4o",
#     api_key=OPENAI_API_KEY
# )

## Ollama local text-only LLM model
llm = ChatOllama(model="tinyllama") 

## Ollama local vision-capable model
#
# llm = ChatOllama(
#     model="openbmb/minicpm-v4"
# )

# =========================================================
# 2. IMAGE -> BASE64 ENCODING
# =========================================================
#
# LLM APIs usually do NOT receive raw image files directly.
#
# Image bytes are converted into base64 text format.
#
# Example:
#
# image.jpg
# ->
# "/9j/4AAQSkZJRgABAQAAAQABAAD..."
#
# This base64 string is later embedded into prompt.

def encode_image(image_file):

    # Read uploaded image bytes
    image_bytes = image_file.read()

    # Convert bytes -> base64 bytes
    encoded_bytes = base64.b64encode(image_bytes)

    # Convert bytes -> normal Python string
    return encoded_bytes.decode()


# =========================================================
# 3. MULTIMODAL PROMPT
# =========================================================
#
# Earlier prompts:
#
# ("human", "{input}")
#
# were TEXT-ONLY prompts.
#
# ---------------------------------------------------------
#
# Current prompt:
#
# Human message contains:
#   1. text block
#   2. image block
#
# This is called:
#
# MULTIMODAL PROMPTING
#
# Meaning:
# LLM receives:
#   - text
#   - image
#
# together.

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 

            # Defines assistant role
            "You are a helpful assistant that can describe images."
        ),
        (
            "human",
            [
                # =================================================
                # TEXT BLOCK
                # =================================================
                #
                # User question text
                #
                # Example:
                # "Explain this image"
                {
                    "type": "text", 
                    "text": "{input}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # IMPORTANT:
                        # {image} gets replaced at runtime
                        # inside chain.invoke(...)
                        "url": f"data:image/jpeg;base64,""{image}",

                        # low:
                        #   faster + cheaper
                        #
                        # high:
                        #   more detailed image analysis                        
                        "detail": "low",
                    },
                },
            ],
        ),
    ]
)

# =========================================================
# 4. CREATE LCEL CHAIN
# =========================================================
#
# LCEL pipeline:
#
# prompt
#   ->
# llm
#
# Meaning:
#
# build prompt
#   ->
# send to model
#   ->
# return response

chain = prompt | llm


# =========================================================
# 5. STREAMLIT UI
# =========================================================

st.title("Ask anything")

# Upload image
uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg","png"]
)

# User question
question = st.text_input("Enter a question")


# =========================================================
# 6. MAIN LOGIC
# =========================================================
#
# Run only when:
#   - image uploaded
#   - question entered

if question:

    # Convert uploaded image -> base64 string
    image = encode_image(uploaded_file)

    # Runtime flow:
    #
    # User Question
    #        +
    # Uploaded Image
    #        ↓
    # Image converted to base64
    #        ↓
    # Multimodal Prompt
    #        ↓
    # Vision LLM
    #        ↓
    # Text Response
    response = chain.invoke(
        {
            "input": question,

            # Replaces:
            # {image}
            #
            # inside prompt template            
            "image":image
        }
    )
    st.write(response.content)

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\S11_image-processing
# python -m streamlit run 2_streamlit_images_demo.py 


# =========================================================
# TEST QUESTIONS
# =========================================================
#
# Generic:
# "Explain this image"
#
# Image-specific:
# "What color is the airplane?"
#
# Counting:
# "How many people are visible?"
#
# OCR:
# "What text is written on the sign?"




# =========================================================
# IMPORTANT ARCHITECTURE DIFFERENCE
# =========================================================
#
# This is NOT RAG.
#
# No:
#   - retriever
#   - embeddings
#   - vector DB
#   - Chroma
#
# are involved.
#
# ---------------------------------------------------------
#
# RAG Architecture:
#
# Question
#    ↓
# Retriever
#    ↓
# Context
#    ↓
# LLM
#
# ---------------------------------------------------------
#
# Multimodal Vision Architecture:
#
# Text
#   +
# Image
#    ↓
# Vision LLM
#    ↓
# Answer
