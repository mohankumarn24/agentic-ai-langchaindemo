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
# llm = ChatOllama(model="openbmb/minicpm-v4")


# =========================================================
# 2. IMAGE -> BASE64 ENCODING
# =========================================================

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode()


# =========================================================
# 3. MULTIMODAL PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can verify identification documents."),
        (
            "human",
            [
                {"type": "text", "text": "Verify whether the uploaded identification document matches the provided user details"},
                {"type": "text", "text": "Name: {user_name}"},
                {"type": "text", "text": "DOB: {user_dob}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,""{image}",
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

chain = prompt | llm


# =========================================================
# 5. STREAMLIT UI
# =========================================================

st.title("KYC Verification Application")
st.write("Upload your identification document:")
# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# User question
user_name = st.text_input("Enter your name")
user_dob = st.date_input("Enter your date of birth")

# =========================================================
# 6. MAIN LOGIC
# =========================================================

if uploaded_file is not None and user_name and user_dob:
    st.image(uploaded_file, caption="Uploaded Document", width="stretch")
    st.write("Processing your document...")
    image = encode_image(uploaded_file)
    response = chain.invoke({"user_name": user_name, "user_dob": user_dob, "image": image})
    st.write(response.content)

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\S11_image-processing
# python -m streamlit run 3_kyc_verification.py