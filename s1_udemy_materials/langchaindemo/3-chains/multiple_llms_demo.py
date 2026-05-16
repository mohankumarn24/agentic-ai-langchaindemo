import os
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm1=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
llm2=ChatOllama(model="mistral")
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title.	
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""You need to write a powerful speech of 350 words
     for the following title: {title}
    """
)

first_chain = title_prompt | llm1 | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = speech_prompt | llm2
final_chain = first_chain | second_chain

st.title("Speech Generator")

topic = st.text_input("Enter the topic:")

if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content)