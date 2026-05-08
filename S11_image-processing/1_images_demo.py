import os
import base64

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# =========================================================
# 1. LLM
# =========================================================

## OpenAI LLM
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

## Ollama local text-only LLM model
llm = ChatOllama(model="tinyllama")                     # 'tinyllama' is 'text-only model' model, bit still works. 
                                                        # But, it cannot answer image specific questions: 
                                                        #   - What color is the airplane?
                                                        #   - How many people are in the image?

## Ollama local vision-capable model                                                        
# llm = ChatOllama(model="openbmb/minicpm-v4")          # Vision-capable multimodal model (heavier CPU usage locally)


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

def encode_image(image_path):

    # Open image in binary mode ("rb")
    with open(image_path, "rb") as image_file:

        # Convert image bytes -> base64 bytes
        #
        # Example:
        # b'\\xff\\xd8\\xff...'
        # ->
        # b'/9j/4AAQSk...'

        encoded_bytes = base64.b64encode(
            image_file.read()
        )

        # Convert bytes -> normal Python string
        return encoded_bytes.decode()


# Encode local image
image = encode_image("airport_terminal_journey.jpeg")


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
                {

                # =================================================
                # TEXT BLOCK
                # =================================================
                #
                # User question text
                #
                # Example:
                # "Explain this image"
                                    
                    "type": "text", 
                    "text": "{input}"
                },

                # =================================================
                # IMAGE BLOCK
                # =================================================
                #
                # Sends image to vision-capable model.
                #
                # Image is passed as:
                #
                # data:image/jpeg;base64,...
                #
                # which is standard browser-style
                # base64 image format.
                                
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",

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
# 4. CREATE CHAIN
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
# 5. EXECUTE MULTIMODAL INFERENCE
# =========================================================
#
# Runtime flow:
#
# Text Question
#        +
# Image
#        ↓
# Prompt Construction
#        ↓
# Vision Model
#        ↓
# Image Understanding
#        ↓
# Text Response

response = chain.invoke(
    {
        "input": "Explain this image"
    }
)

# =========================================================
# 6. PRINT FINAL RESPONSE
# =========================================================

print(response.content)

## Run
# ollama pull openbmb/minicpm-v4
# 
# cd D:\dev\github\agentic-ai-langchaindemo\S11_image-processing
# python 1_images_demo.py


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




# =========================================================
# IMPORTANT NOTE
# =========================================================
#
# TinyLlama:
# - may generate generic image descriptions
# - but does NOT actually understand image pixels
#
# MiniCPM-V4:
# - truly processes image content
# - can answer image-specific questions