from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="deepseek-r1:1.5b")

question = input("Enter question: ")
response = llm.invoke(question)
print(response.content)

# DeepSeek reasoning models sometimes internally generate
# <think>
#   step-by-step reasoning
# </think>

## Run
# cd D:\dev\github\agentic-ai-langchaindemo\s14_use-deepseek-models
# python 1_deepseek_demo.py