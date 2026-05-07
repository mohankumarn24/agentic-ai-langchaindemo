import os
import streamlit as st
import uuid

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

## 1. OpenAI Cloud API key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

## 2. Ollama local
llm_ollamaLocal = OllamaLLM(model="tinyllama")

## 3. Ollama cloud API key
llm_ollamaCloud = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"
    }
)

## PROMPT TEMPLATE
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # Defines AI behavior/personality
            "You are an Agile Coach. "
            "Answer questions related to Agile process clearly."
        ),

        # Previous chat history gets inserted here
        # Without this, AI will forget earlier conversation
        # MessagesPlaceholder is used for chat history injection
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            # Current user question
            "{input}"
        )
    ]
)

# LCEL CHAIN
# chain = prompt_template | llm_ollamaLocal | StrOutputParser()
chain = prompt_template | llm_ollamaCloud | StrOutputParser()

# Temporary in-memory conversation storage
# Stores previous user + AI messages temporarily. Initially empty ie., []
"""
[
    HumanMessage(content="Explain Scrum"),
    AIMessage(content="Scrum is an Agile framework..."),

    HumanMessage(content="Summarize this"),
    AIMessage(content="Scrum is a lightweight Agile framework...")
]
"""
history_for_chain = InMemoryChatMessageHistory()

# Adds conversational memory support to chain
# Wrapper that automatically:
#  - Retrieves old messages
#  - Injects memory into prompt
#  - Stores latest conversation
chain_with_history = RunnableWithMessageHistory(
    # Original LCEL chain
    chain,

    # Function that returns chat history object
    #  - session_id helps separate conversations
    #  - Different session_id -> different chat history
    lambda session_id: history_for_chain,

    # The current user question is stored inside dictionary key called "input"
    # invoked inside 'chain_with_history.invoke' ie ' "input": question '
    input_messages_key="input",

    # Variable name used in MessagesPlaceholder
    # Inject old conversation messages into variable called "chat_history"
    # Because prompt contains: MessagesPlaceholder(variable_name="chat_history")
    history_messages_key="chat_history"
)

#   def get_history(session_id):
#        return history_for_chain
# 
# is same as:
#   lambda session_id: history_for_chain

print("Agile Guide")

session_id = str(uuid.uuid4())
while True:
    # Debugging: Prints stored conversation memory: (previous HumanMessage + AIMessage objects)
    # print(history_for_chain.messages)

    question = input("\nEnter the question: ")

    if question.lower() == "exit":
        print("\nExiting chatbot. Good Bye!!")
        break

    if question:
        # invoke() executes the chain
        response = chain_with_history.invoke(
            # Current user input
            {
                "input": question
            },

            # Runtime configuration
            {
                "configurable": 
                {
                    # Unique conversation/session identifier
                    #  - Same session_id       -> remembers old chat
                    #  - Different session_id  -> fresh conversation
                    "session_id": session_id
                    # "session_id": "abc123"
                }
            }
        )

        print("\nAI Response:")
        print(response)
        print("-" * 50)

## Run:
# cd D:\dev\github\agentic-ai-langchaindemo\s7_maintaining-chathistory
# python 2_chathistory_demo.py
# http://localhost:8501/          

# First input   : Explain scrum
# First output  : Scrum is a lightweight, evidence‑based framework for building complex products. It focuses on ...

# Second input  : can you summarise this in two sentences
# second output : Scrum is an evide... It uses...


## Real output after enabling debugging:
# [HumanMessage(content='what is scrum', additional_kwargs={}, response_metadata={}), 
# AIMessage(content='### Scrum – the Agile framework that turns chaos into predictability\n\n> *Scrum is a lightweight, iterative approach to complex product development. It helps teams deliver value faster, learn quickly, and stay aligned with customer needs.*\n\n---\n\n## 1. Why Scrum?  \n| Problem | Scrum Solution |\n|---------|----------------|\n| **Unclear vision & scope** | Sprint Planning & Product Backlog give incremental direction. |\n| **Long, opaque development cycles** | Short sprints (1–4\u202fweeks) and fixed cadence deliver early, measurable results. |\n| **Poor communication** | Daily Scrum & reviews keep everyone on the same page. |\n| **Slow responses to change** | Incremental delivery + frequent retrospectives embed learning and adaptation. |\n\n---\n\n## 2. Core Ingredients\n\n| Element | What it is | Key Benefit |\n|---------|------------|-------------|\n| **Roles** | • **Product Owner** – owns the vision & prioritizes the backlog.<br>• **Scrum Master** – facilitator & servant leader.<br>• **Development Team** – cross‑functional, self‑organizing. | Clear responsibilities, reduced friction. |\n| **Artifacts** | • **Product Backlog** – dynamic, ordered list of everything needed. <br>• **Sprint Backlog** – work committed for the sprint. <br>• **Increment** – usable, potentially shippable deliverable. | Transparent progress, focused scope, measurable value. |\n| **Events** | • **Sprint Planning** – set goal & plan tasks.<br>• **Daily Scrum** – inspect, adapt, coordinate.<br>• **Sprint Review** – show the Increment, gather feedback.<br>• **Sprint Retrospective** – learn & improve. | Structured rhythm, regular feedback loops. |\n| **Definition of Done (DoD)** | A shared checklist of what “done” means for the team. | Quality and predictability. |\n\n---\n\n## 3. The Scrum Lifecycle (simplified)\n\n```mermaid\nflowchart TD\n  PB[[Product Backlog]]\n  PPlanning([Sprint Planning]) --> SB([Sprint Backlog])\n  SB --> DSS([Daily Scrum]) --> Devs[Development]\n  Devs --> SR([Sprint Review]) --> PR([Sprint Retrospective])\n  PR --> PB\n```\n\n1. **Product Owner** refines the backlog → **Sprint Planning** sets sprint goal & commits items → **Daily Scrums** keep the team coordinated → Working **Increment** delivered → **Sprint Review** + **Retrospective** produce learnings → Loop.\n\n---\n\n## 4. Scrum in Practice\n\n| Common Tool | How to Use It |\n|-------------|---------------|\n| **Kanban Board** | Visualize backlog items (To Do → In Progress → Done). |\n| **Story Points** | Estimate effort relative to others (instead of hours). |\n| **Velocity** | Measure average points completed per sprint; helps forecasting. |\n| **Burndown / Burnup Chart** | Track sprint progress; highlights risks early. |\n\n---\n\n## 5. Typical Questions\n\n| Question | Short Answer |\n|----------|--------------|\n| *What’s the difference between Scrum and Kanban?* | Scrum has time‑boxed sprints and prescribed roles; Kanban is continuous, no defined cadence. |\n| *Can non‑software teams use Scrum?* | Yes – the core concepts (roles, ceremonies, incremental value) apply to any complex work. |\n| *Is Scrum rigid?* | No – its lightweight structure adapts; teams tailor ceremonies and artifacts to fit context while preserving transparency. |\n| *How do you start a new Scrum team?* | 1. Get a single, cross‑functional team. 2. Assign roles. 3. Create an initial Product Backlog. 4. Run your first Sprint Planning and Sprint. 5. Iterate and refine. |\n\n---\n\n## 6. Quick Reference Cheat‑Sheet\n\n| **Ceremony** | **Timebox** | **Purpose** |  \n|--------------|------------|-------------|  \n| Sprint Planning | 2–4\u202fhrs per 1‑month sprint | Define sprint goal & select backlog items. |  \n| Daily Scrum | 15\u202fmin | Inspect progress, adapt plan, coordinate. |  \n| Sprint Review | 1–2\u202fhrs | Inspect Increment with stakeholders, gather feedback. |  \n| Sprint Retrospective | 1–2\u202fhrs | Reflect on process, identify improvements. |  \n\n---\n\n**Bottom line:**  \nScrum is a *structured, evidence‑based framework* that transforms complex product work into short, transparent iterations. By empowering teams to plan, inspect, and adapt continuously, it delivers higher quality, faster time‑to‑market, and a better alignment with customer value. 🚀', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), 
# HumanMessage(content='summarize in one sentence', additional_kwargs={}, response_metadata={}), 
# AIMessage(content='Scrum is a lightweight, iterative Agile framework that uses a fixed‑length Sprint, clearly defined roles, tangible artifacts, and regular ceremonies to deliver valuable increments, foster continuous improvement, and maintain transparent, cross‑functional collaboration.', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[])]