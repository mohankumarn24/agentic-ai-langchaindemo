import os
import whisper
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

def get_transcription(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

def process_transcription(transcription, query):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that can interpret and explain audio transcriptions."),
            (
                "human",
                [
                    {"type": "text", "text": "{input}"}
                ]
            )
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"input": transcription + "\n\nQuery: " + query})

    return response.content

st.title("Audio Transcription and Query Assistant")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:

    audio_path = "uploaded_audio" + os.path.splitext(uploaded_file.name)[1]
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    transcription = get_transcription(audio_path)
    st.write("Transcription:")
    st.write(transcription)

    query = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        if query:
            answer = process_transcription(transcription, query)
            st.write("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a query.")
else:
    st.info("Please upload an audio file.")
