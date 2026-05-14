from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="deepseek-r1:1.5b")

question = input("Enter question: ")
response = llm.invoke(question)

## Optional: If using ChatOllama
# response = llm.invoke([
#     (
#         "system", 
#         "You are a factual assistant. Answer in one sentence. Do not invent details. If unsure, say I don't know."),
#     (
#         "human", question
#     )
# ])

print(response.content)

## Run
# TODO: May stress CPU
# cd D:\dev\github\agentic-ai-langchaindemo\s14_use-deepseek-models
# python 1_deepseek_demo.py


## Output:
# DeepSeek reasoning models sometimes internally generate <think>.
# But, DeepSeek may not always show it
#
#   <think>
#       step-by-step reasoning
#   </think>
# 
# Questions to ask:
# Q: If a train travels 60 km in 1.5 hours, what is its speed?
# Q: Where is Taj Mahal?