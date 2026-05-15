from langchain_openai import ChatOpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

product_prompt = PromptTemplate(
    input_variables=["product_name", "features"],
    template="""
    You are an experienced marketing specialist. 
    Create a catchy subject line for a marketing 
    email promoting the following product: {product_name}. 
    Highlight these features: {features}. 
    Respond with only the subject line.
    """
)

email_prompt = PromptTemplate(
    input_variables=["subject_line", "product_name", "target_audience"],
    template="""
    Write a marketing email of 300 words for the 
    product: {product_name}. Use the subject line:
     {subject_line}. Tailor the message for the 
     following target audience: {target_audience}.
      Format the output as a JSON object with three 
      keys: 'subject', 'audience', 'email' and fill 
      them with respective values.
    """
)

first_chain = product_prompt | llm | StrOutputParser()
second_chain = email_prompt | llm | JsonOutputParser()

overall_chain = (
    first_chain |
    (lambda subject_line: {"subject_line": subject_line, "product_name": product_name, "target_audience": target_audience}) |
    second_chain
)

st.title("Marketing Email Generator")

product_name = st.text_input("Input Product Name")
features = st.text_input("Input Product Features (comma-separated)")
target_audience = st.text_input("Input Target Audience")

if product_name and features and target_audience:
    response = overall_chain.invoke({"product_name": product_name, "features": features})
    st.write(response)