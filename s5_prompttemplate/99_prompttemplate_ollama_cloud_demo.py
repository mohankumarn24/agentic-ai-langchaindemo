import os
import streamlit as st

from ollama import Client
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# OLLAMA CLOUD CLIENT
client = Client(
    host="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

# STREAMLIT UI
st.title("AI Speech Generator")

topic = st.text_input("Enter speech topic")
emotion = st.text_input("Enter emotion")

# PROMPTS
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an experienced speech writer.
    Create one powerful speech title for the following topic:
    Topic: {topic}
    Return ONLY the title.
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""
    Write a powerful {emotion} speech of around 350 words.
    Title: {title}
    Return output STRICTLY in valid JSON format:
    {{
        "title": "...",
        "speech": "..."
    }}
    """
)

# GENERATE
if st.button("Generate Speech"):
    
    if not topic or not emotion:
        st.warning("Please enter both topic and emotion.")
    else:
        # STEP 1 -> Generate title
        formatted_title_prompt = title_prompt.format(
            topic=topic
        )

        title_response = client.chat(
            model="gpt-oss:20b",
            messages=[
                {
                    "role": "user",
                    "content": formatted_title_prompt
                }
            ]
        )

        title = title_response["message"]["content"].strip()

        st.subheader("Generated Title")
        st.write(title)

        # STEP 2 -> Generate speech
        formatted_speech_prompt = speech_prompt.format(
            title=title,
            emotion=emotion
        )

        speech_response = client.chat(
            model="gpt-oss:20b",
            messages=[
                {
                    "role": "user",
                    "content": formatted_speech_prompt
                }
            ]
        )

        raw_output = speech_response["message"]["content"]

        # PARSE JSON
        parser = JsonOutputParser()

        try:
            parsed_output = parser.parse(raw_output)
            st.subheader(parsed_output["title"])
            st.write(parsed_output["speech"])
        except Exception as e:
            st.error("JSON Parsing Failed")
            st.write(raw_output)
            st.write(str(e))

# -------------------------------------------------
# RUN
# -------------------------------------------------
# Windows CMD:
# set OLLAMA_API_KEY=your_api_key
#
# PowerShell:
# $env:OLLAMA_API_KEY="your_api_key"
#
# Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s5_prompttemplate
# python -m streamlit run 99_prompttemplate_ollama_cloud_demo.py
# http://localhost:8501/  