import os
from typing import TypedDict, Annotated
from operator import add

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()
# LOAD ENVIRONMENT VARIABLES



api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Please set your GROQ_API_KEY in the .env file.")
    st.stop()



# ============================================================
# SETUP LLM
# ============================================================

llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-20b",
    api_key=api_key
)


# ============================================================
# STATE SCHEMA
# ============================================================

class CodeState(TypedDict):
    query: str
    context: str
    messages: Annotated[list, add]
    response: str


# ============================================================
# HELPER FUNCTION
# ============================================================

def analyze_code_context(query: str) -> str:
    """Prepare code for analysis."""
    return f"Analyzing the following code snippet:\n\n{query}"


# ============================================================
# TOOL
# ============================================================

@tool
def get_code_analysis_guidance(query: str) -> str:
    """Provide guidance for analyzing source code."""
    
    return (
        "Analyze the code for syntax errors, logical errors, "
        "runtime issues, performance problems, readability, "
        "security concerns, and best practices."
    )


tools = [get_code_analysis_guidance]


# ============================================================
# LANGGRAPH NODE 1 - CREATE PROMPT
# ============================================================

def create_prompt_node(state: CodeState) -> CodeState:

    query = state["query"]

    context = analyze_code_context(query)

    prompt = f"""
You are an expert code analysis assistant.

Analyze the user's code carefully.

Context:
{context}

Your task:

1. Identify syntax errors.
2. Identify logical errors.
3. Identify possible runtime errors.
4. Check whether required libraries are missing.
5. Explain what is wrong in simple language.
6. Provide corrected code when necessary.
7. Suggest improvements and best practices.
8. Explain the important changes clearly.

User Code:
{query}

Give a clear and useful answer.
"""

    return {
        "query": query,
        "context": context,
        "messages": [HumanMessage(content=prompt)],
        "response": ""
    }


# ============================================================
# LANGGRAPH NODE 2 - GENERATE RESPONSE
# ============================================================

def generate_response_node(state: CodeState) -> CodeState:

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    result = agent.invoke(
        {
            "messages": state["messages"]
        }
    )

    if result and "messages" in result:

        bot_message = result["messages"][-1]

        response_text = bot_message.content

        # Handle cases where content is returned as a list
        if isinstance(response_text, list):
            response_text = "\n".join(
                str(item) for item in response_text
            )

    else:
        response_text = "No response generated."

    return {
        "query": state["query"],
        "context": state["context"],
        "messages": [
            AIMessage(content=response_text)
        ],
        "response": response_text
    }


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_code_graph():

    graph = StateGraph(CodeState)

    graph.add_node(
        "create_prompt",
        create_prompt_node
    )

    graph.add_node(
        "generate_response",
        generate_response_node
    )

    graph.add_edge(
        START,
        "create_prompt"
    )

    graph.add_edge(
        "create_prompt",
        "generate_response"
    )

    graph.add_edge(
        "generate_response",
        END
    )

    return graph.compile()


code_workflow = build_code_graph()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Code Analyzer Chatbot",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Code Analyzer Chatbot")
st.write("LangGraph + Groq LLM")


# ============================================================
# SESSION STATE
# ============================================================

if "conversation" not in st.session_state:
    st.session_state.conversation = []


# ============================================================
# CODE INPUT
# ============================================================

user_code = st.text_area(
    "Paste your code snippet here:",
    height=300,
    placeholder="Paste your Python, Java, C, SQL, or other code here..."
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze Code"):

    if not user_code.strip():

        st.warning("Please paste some code first.")

    else:

        try:

            with st.spinner("Analyzing code..."):

                initial_state = {
                    "query": user_code,
                    "context": "",
                    "messages": [],
                    "response": ""
                }

                result = code_workflow.invoke(
                    initial_state
                )

            bot_response_text = result.get(
                "response",
                "No response generated."
            )

            # Save conversation
            st.session_state.conversation.append(
                {
                    "user": user_code,
                    "bot": bot_response_text
                }
            )

            st.success("Analysis Complete ✅")

            # ==================================================
            # DISPLAY USER CODE
            # ==================================================

            st.subheader("Your Code")

            st.code(
                user_code,
                language="python"
            )

            # ==================================================
            # DISPLAY AI RESPONSE
            # ==================================================

            st.subheader("Assistant Analysis")

            st.markdown(
                bot_response_text
            )

            # ==================================================
            # DISPLAY CONTEXT
            # ==================================================

            with st.expander("View Context Used"):

                st.text(
                    result.get(
                        "context",
                        "No context found."
                    )
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

            with st.expander("View Detailed Error"):

                import traceback

                st.code(
                    traceback.format_exc()
                )


# ============================================================
# CONVERSATION HISTORY
# ============================================================

if st.session_state.conversation:

    st.divider()

    st.subheader("📝 Conversation History")

    for index, turn in enumerate(
        st.session_state.conversation,
        start=1
    ):

        st.markdown(
            f"### Analysis {index}"
        )

        st.markdown("**You:**")

        st.code(
            turn["user"],
            language="python"
        )

        st.markdown("**Assistant Analysis:**")

        st.markdown(
            turn["bot"]
        )

        st.divider()