import os
import base64
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.globals import set_debug
from langchain_ollama import ChatOllama

set_debug(True)

## API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_api_key = os.getenv("OLLAMA_API_KEY")

# ----------------------------------------
# 1. LLM setup (gpt-4o for vision + tools)
# ----------------------------------------
# Vision model: must support images
vision_llm = ChatOpenAI(
    model="gpt-5-nano", 
    api_key=openai_api_key, 
    temperature=0                                           # Controls randomness: 0 = deterministic/focused, higher values = more creative/random
)
    
agent_llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="https://ollama.com",
    headers={
        # Adds API key to request headers. Example: Authorization: Bearer xxxxx
        "Authorization": f"Bearer {ollama_api_key}"
    },
    temperature=0
)

# -------------------------
# 2. Helper to encode image
# -------------------------
def encode_image(image_file):
    image_file.seek(0)                                                  # Go to beginning
    image_b64 = base64.b64encode(image_file.read()).decode("utf-8")     # Read full image, convert to base64
    mime_type = image_file.type
    return image_b64, mime_type

# ------------------------------------
# 3. Vision prompt (identify landmark)
# ------------------------------------
vision_prompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You identify famous landmarks from images. Return only the landmark name. If unsure, say Unknown."
    ),
    (
        "human",
        [
            {
                "type": "text", 
                "text": "Identify the landmark in this image. Return only the landmark name."
            },
            {
                "type": "image_url",
                "image_url": {
                    # The uploaded image is converted to a base64 string.
                    # LangChain replaces {mime_type} and {image} at runtime.
                    #
                    # Example:
                    #   mime_type = "image/jpeg"
                    #   image = "/9j/4AAQSkZJRgABAQAAAQ..."
                    #
                    # Then:
                    #   "data:{mime_type};base64,{image}"
                    #
                    # becomes:
                    #   "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
                    #
                    # This data URL is sent to the vision model as the image input.
                    "url": "data:{mime_type};base64,{image}",
                    "detail": "low",
                },
            },
        ],
    ),
])

vision_chain = vision_prompt | vision_llm

# ------------------------------
# 4. Tools (Wikipedia + DuckDuckGo)
# ------------------------------
tools = load_tools(["wikipedia", "ddg-search"])
tool_names = ", ".join([tool.name for tool in tools])

# ------------------------------
# 5. ReAct-style agent (new v1 API)
# ------------------------------
react_system_prompt = f"""
                      You are a helpful ReAct-style landmark information agent.

                      You have access to these tools:
                      {tool_names}

                      Use search when the answer needs factual, historical, or current web information.
                      If you use a tool, base your final answer on the tool result.

                      Give a clear and concise answer.
                      """

agent = create_agent(
    model=agent_llm,
    tools=tools,
    system_prompt=react_system_prompt
)

# ---------------
# 6. Streamlit UI
# ---------------
st.title("Landmark Helper - Vision + Agent")

uploaded_file = st.file_uploader("Upload landmark image", type=["jpg", "jpeg", "png"])
question = st.text_input("Enter a question about the landmark")

if st.button("Ask") and uploaded_file and question:
    try:
        # First: use vision chain to get landmark name
        with st.spinner("Identifying landmark..."):
            image_b64, mime_type = encode_image(uploaded_file)
            vision_response = vision_chain.invoke({
                "image": image_b64,
                "mime_type": mime_type
            })
            landmark_name = vision_response.content.strip()

        st.write("Detected landmark:", landmark_name)

        if landmark_name.lower() == "unknown":
            st.warning("Could not confidently identify the landmark.")
            st.stop()

        agent_task = f"""
                     Landmark: {landmark_name}
                     Question: {question}

                     Answer the question clearly and briefly.
                     """

        # Then: send combined task to tools agent
        with st.spinner("Searching and answering..."):
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": agent_task,
                        }
                    ]
                }
            )

        final_msg = result["messages"][-1]
        st.write(final_msg.content)
    except Exception as e:
        st.error("Failed while processing the image or running the agent.")
        st.exception(e)

## Run
# ollama pull qwen3.5:4b
#
# cd D:\dev\github\agentic-ai-langchaindemo\s12_agents
# python -m streamlit run 2_landmark_helper.py
# http://localhost:8501/  

## Output:
# Browse file: 
#     D:\dev\github\agentic-ai-langchaindemo\s12_agents\statue_of_liberty.jpg
#     D:\dev\github\agentic-ai-langchaindemo\s12_agents\taj_mahal.jpg
#
# Enter question: 
#     Where is it located?
#
# Response: 
#     Detected landmark: Taj Mahal
#     Taj Mahal is located in Agra, Uttar Pradesh, India.