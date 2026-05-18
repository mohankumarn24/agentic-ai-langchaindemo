#######################################################################
# 1. Simple Prompt
#######################################################################
prompt = f"Answer briefly.\nQuestion: {question}\nAnswer:"
response = llm.invoke(prompt)

#######################################################################
# 2. Prompt Template
#######################################################################
prompt_template = PromptTemplate(
    input_variables=["country", "no_of_paragraphs", "language"],
    template="""
    You are an expert in traditional cuisines.

    If the country is fictional or non-existent, answer exactly:
    I don't know.

    Question:
    What is the traditional cuisine of {country}?

    Instructions:
    - Answer in {no_of_paragraphs} short paragraphs.
    - Use {language}.
    - Keep the answer factual and concise.
    """
)

response = llm.invoke(
    prompt_template.format(country=country,
        no_of_paragraphs=no_of_paragraphs,
        language=language)
)
                                    
#######################################################################
# 3a. LCEL
#######################################################################
prompt_template = PromptTemplate(
    input_variables=["city", "month", "language", "budget"],
    template="""
    You are a helpful travel guide.

    Create a practical travel guide for {city} for someone visiting in {month}.

    Include:
    1. Must-visit attractions.
    2. Local cuisine the traveler should try.
    3. Useful phrases in {language}.
    4. Tips for traveling on a {budget} budget.

    Keep the answer clear, useful, and beginner-friendly.
    """
)

chain = prompt_template | llm

response = chain.invoke({
    "city": city,
    "month": month,
    "language": language,
    "budget": budget
})                                    


# 3b. Sequential chain
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    You are an experienced speech writer.

    Craft an impactful title for a speech on the following topic:
    {topic}

    Answer exactly with one title.
    """
)

speech_prompt = PromptTemplate(
    input_variables=["title"],
    template="""
    Write a powerful speech of around 350 words for the following title:

    {title}
    """
)

first_chain = title_prompt | title_llm | StrOutputParser() | RunnableLambda(display_title_and_return_title)
second_chain = speech_prompt | speech_llm | StrOutputParser()
final_chain = first_chain | second_chain
response = final_chain.invoke({
    "topic": topic
})

#######################################################################
# 4. Agents
#######################################################################
tools = load_tools(["wikipedia", "ddg-search"])
tool_names = ", ".join([tool.name for tool in tools])                                       # "wikipedia, ddg-search"
react_system_prompt = """
                      You are a helpful ReAct-style AI agent.
					  
                      You have access to these tools:
                      {tool_names}

                      Use tools only when needed.
                      Use Wikipedia for encyclopedia-style facts.
                      Use search for current or web-based information.

                      If you use a tool, base your final answer on the tool result.
                      Give a clear final answer to the user.
                      """
                      
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt
)

task = st.text_input("Who is the current CEO of Microsoft?")
result = agent.invoke({
    "messages": [
        {
            "role": "user",                             
            "content": task                                                         
        }
    ]
})

# | Role        | Meaning                   | Example                                      |
# | ----------- | ------------------------- | -------------------------------------------- |
# | `system`    | Instruction to the AI     | `"You are a helpful assistant"`              |
# | `user`      | Human/user message        | `"Who is CEO of Microsoft?"`                 |
# | `assistant` | AI’s previous reply       | `"Satya Nadella is CEO"`                     |
# | `tool`      | Result returned by a tool | Search result, calculator output, API result |

# messages = [
#     {
#         "role": "system",                                                                 # System instruction:
#         "content": "You are a helpful assistant."                                         # You are a helpful assistant.
#     },
#     {
#         "role": "user",                                                                   # User asks:
#         "content": "Who is the current CEO of Microsoft?"                                 # Who is the current CEO of Microsoft?
#     }
# ]


#######################################################################
# 5. RAG with history - Retrieval-Augmented Generation
#######################################################################
document = TextLoader("product-data.txt", encoding="utf-8").load()                          # Load text file as LangChain Document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)          # Split large document into overlapping chunks
chunks = text_splitter.split_documents(document)                                            # Create smaller chunks for embedding

# Create Vector DB for semantic search
# Chroma stores embeddings and allows semantic search
persist_directory = "chroma_db_openai"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print("Loading existing Chroma DB...")
    vector_store = Chroma(
        embedding_function=openai_embeddings_cloud,
        persist_directory=persist_directory
    )
else:
    print("Creating new Chroma DB...")
    vector_store = Chroma.from_documents(
        documents=chunks,                                                                   # List of documents to add to the VectorStore                                    
        embedding=openai_embeddings_cloud,
        persist_directory=persist_directory
    )
    
# Retriever performs semantic search on vector DB later    
retriever = vector_store.as_retriever()                                                     # Convert vector DB into retriever for semantic search

# store past history
history_for_chain = StreamlitChatMessageHistory()                                           # Store chat messages in Streamlit session

# chat_history + input → reframe question → retrieve docs
# Prompt to rewrite follow-up question into standalone question
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that reformulates follow-up
            questions into standalone questions.

            Use the chat history and the latest user input to create
            a self-contained question.

            Do NOT answer the question, only rewrite it.
            """
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human", 
            "{input}"
        ),
    ]
)


# retrieved docs (context) + chat_history + input → final answer
# Prompt to answer using retrieved docs + chat history + current input
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an assistant for answering questions.
            Use the provided context to respond.
            If the answer isn't clear from the context, say you don't know.
            Limit your response to three concise sentences.

            Context:
            {context}
            """
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human", 
            "{input}"
        ),
    ]
)   

history_aware_retriever = create_history_aware_retriever(llm_openai_cloud, retriever, contextualize_q_prompt)   # Rewrites input using chat_history, then retrieve relevant docs
qa_chain = create_stuff_documents_chain(llm_openai_cloud, qa_prompt)                                            # Generate final answer using retrieved docs as context + chat_history + input
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)                                           # Connect retriever and answer chain

chain_with_history = RunnableWithMessageHistory(                                                                # Add automatic chat memory to RAG chain
    rag_chain,                                                                                                  # Chain being wrapped with memory
    lambda session_id : history_for_chain,                                                                      # Function that returns chat history for a session_id
    input_messages_key="input",                                                                                 # The key where the current user question is found  
    history_messages_key="chat_history",                                                                        # The key used to inject previous messages into the wrapped chain
    output_messages_key="answer",                                                                               # The key where the final AI answer is found in rag_chain output
)

response = chain_with_history.invoke(                                                                           # Run full conversational RAG pipeline
    {
        "input": question
    },
    {
        "configurable":
            {
                "session_id": session_id
            }
    }
)

#######################################################################
# 6a. MCP
mcp = FastMCP(name="Tool Server")

# Register this function as an MCP tool
@mcp.tool()
def wikipedia_search(query: str) -> str:
    try:
        # Get 2-sentence summary from Wikipedia
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        # Return error message if Wikipedia search fails
        return f"Error: {str(e)}"

# Register this function as another MCP tool
@mcp.tool()
def ddg_search(query: str) -> str:
    try:
        # Create DuckDuckGo search client
        with DDGS() as ddgs:
            # Get top 3 text search results
            results = ddgs.text(query, max_results=3)

            # Extract only result body text and join into one string
            return "\n".join([r["body"] for r in results])
    except Exception as e:
        # Return error message if DuckDuckGo search fails
        return f"Error: {str(e)}"
        
mcp.run(transport="streamable-http")
# mcp.run(transport="stdio")


# 6b. MCP
client_streamable = MultiServerMCPClient(
    {
        "tools": 
        {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable-http"
        }
    }
)

# Fetch tools exposed by the MCP server
tools = asyncio.run(client_streamable.get_tools())

agent = create_agent(llm, tools) 
response = asyncio.run(
    agent.ainvoke(
        {
            "messages": task
        }
    )
)

# 6c. MCP
client_stdio = MultiServerMCPClient(
    {
        "tools": 
        {
            "command": "python",
            "args": ["1_mcp_server.py"],
            "transport": "stdio"
        }
    }
)

# Fetch tools exposed by the MCP server
tools = asyncio.run(client_stdio.get_tools())

agent = create_agent(llm, tools)
response = asyncio.run(
    agent.ainvoke(
        {
             "messages": task
        }
    )
)
#######################################################################






# =========================================================
# HISTORY-AWARE RETRIEVER PROMPT
# =========================================================
# Purpose:
# Convert follow-up questions into standalone questions before retrieval.
#
# Why needed:
# User may ask short follow-up questions like:
#   "What about battery?"
#   "Is it waterproof?"
#   "What about save method?"
#
# These questions are unclear without previous chat history.
# This prompt uses:
#   1. chat_history  -> previous conversation context
#   2. input         -> latest user question
#
# It rewrites the latest input into a complete standalone question.
#
# Example:
#   chat_history:
#       Human: Tell me about iPhone 15.
#       AI: iPhone 15 has a 6.1-inch display and A16 chip.
#
#   input:
#       What about battery?
#
#   rewritten question:
#       What is the battery life of iPhone 15?
#
# Important:
# This prompt does NOT answer the question.
# It only rewrites the question for better document retrieval.
#
# Flow:
#   chat_history + input
#          ↓
#   standalone question
#          ↓
#   retriever searches vector DB using standalone question
# =========================================================




# =========================================================
# QA PROMPT
# =========================================================
# Purpose:
# Generate the final answer using retrieved documents.
#
# Why needed:
# After the retriever finds relevant chunks from the vector DB,
# those chunks are passed into this prompt as {context}.
#
# This prompt uses:
#   1. context       -> retrieved document chunks
#   2. chat_history  -> previous conversation context
#   3. input         -> latest user question
#
# Example:
#   context:
#       iPhone 15 provides up to 20 hours of video playback.
#       It supports fast charging up to 50% in around 30 minutes.
#
#   chat_history:
#       Human: Tell me about iPhone 15.
#       AI: iPhone 15 has a 6.1-inch display and A16 chip.
#
#   input:
#       What about battery?
#
#   final answer:
#       iPhone 15 provides up to 20 hours of video playback.
#       It supports fast charging up to 50% in around 30 minutes.
#
# Important:
# This prompt DOES answer the question.
# It should answer only from the provided context.
# If the answer is not available in context, it should say it does not know.
#
# Flow:
#   retrieved docs as context + chat_history + input
#          ↓
#   final answer
# =========================================================