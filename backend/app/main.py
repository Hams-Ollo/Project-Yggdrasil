# main.py - This is the main file for the backend application. This is a multi agentic chatbot assistant which utilizes langraph for managing the conversation state.
#-------------------------------------------------------------------------------------#
# SETUP:
#
# Setup venv and install the requirements
# 1. Create a virtual environment -> python -m venv venv
# 2. Activate the virtual environment -> .\venv\Scripts\Activate
# 3. Install the requirements -> pip install -r requirements.txt
# 4. Run the streamlit app -> streamlit run app.py / streamlit run frontend/streamlit_app.py
#
# 5. Push to github -> git push -u origin main
#-------------------------------------------------------------------------------------#
print("🚀 Starting application...")

# Importing the necessary libraries
import os
import sys
import pydantic
from dotenv import load_dotenv
print("🔑 Loading environment variables...")
load_dotenv()
from typing import Annotated, Literal, TypedDict, List, Dict, operator
import pandas as pd
import numpy as np

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
import functools
from langsmith import traceable
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_core.runnables import RunnableConfig
import datetime

print("📚 All libraries imported successfully")

#-------------------------------------------------------------------------------------#
#  AI Multi-Agent Setup
#-------------------------------------------------------------------------------------#

# Add these imports at the top with the other imports
# from langsmith import traceable
# Remove: from langchain.callbacks.traceable_chain import TraceableChain
# from langchain_core.tracers import ConsoleCallbackHandler
# from langchain.callbacks.manager import CallbackManager
# from langchain_core.runnables import RunnableConfig
# import datetime

# Update LangSmith setup
print("🔄 Setting up LangSmith tracing...")
try:
    from langsmith import Client
    client = Client()
    
    # Configure LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = "Project-Yggdrasil"
    
    # Create callback manager for tracing
    callback_manager = CallbackManager([ConsoleCallbackHandler()])
    print("✅ LangSmith configuration loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: LangSmith setup failed: {str(e)}")
    callback_manager = CallbackManager([])

# Initialize the LLM with callbacks and fallback
print("🤖 Initializing LLMs...")
def get_llm():
    """Get LLM with fallback options"""
    try:
        # Try Groq first
        primary_llm = ChatGroq(
            temperature=0.1,
            model_name="mixtral-8x7b-32768",
            api_key=os.getenv("GROQ_API_KEY"),
            callbacks=callback_manager.handlers
        )
        # Test the LLM
        primary_llm.invoke("test")
        print("✅ Groq LLM initialized successfully")
        return primary_llm
    except Exception as e:
        print(f"⚠️ Groq LLM failed: {str(e)}")
        try:
            # Fallback to OpenAI 4o-mini
            from langchain_openai import ChatOpenAI
            fallback_llm = ChatOpenAI(
                temperature=0.1,
                model_name="gpt-4o-mini",  # Using 4o-mini as fallback
                api_key=os.getenv("OPENAI_API_KEY"),
                callbacks=callback_manager.handlers
            )
            print("✅ Fallback to OpenAI 4o-mini successful")
            return fallback_llm
        except Exception as e:
            print(f"❌ All LLM options failed: {str(e)}")
            raise Exception("No available LLM services")

# Initialize LLM with fallback
try:
    llm = get_llm()
except Exception as e:
    print(f"❌ Fatal error: {str(e)}")
    sys.exit(1)

# Define the agent state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    input: str

print("🎯 Setting up agent prompts...")
# Create prompts for each agent
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing supervisor for a multi-agent system. Your sole responsibility is to analyze user requests and direct them to the most appropriate specialist agent:

RESEARCHER (for information needs):
- Information gathering and analysis
- Fact-checking and verification
- Current events and trends
- Technical explanations

WRITER (for content needs):
- Content creation and editing
- Documentation and guides
- Creative writing
- Professional communications

CODER (for technical needs):
- Code implementation
- Debugging and troubleshooting
- Technical problem-solving
- Programming explanations

Decision Rules:
1. If the request is about finding, analyzing, or explaining information → RESEARCHER
2. If the request is about creating, editing, or structuring content → WRITER
3. If the request involves code, programming, or technical implementation → CODER
4. For general queries about the system → Engage directly
5. For unclear requests → Ask for clarification

For general queries, engage in conversation.
For unclear requests, ask for clarification.

Examples:
"What is quantum computing?" → researcher
"Write a blog post" → writer
"Debug this code" → coder
"Hello" → [engage in conversation]
"Not sure what I need" → [ask for clarification]"""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research specialist. Focus on finding and analyzing information."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writing specialist. Focus on creating and editing content."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a coding specialist. When asked a coding question:
    1. Provide a clear, working solution with code examples
    2. Explain how the code works
    3. Include any relevant best practices or considerations
    Always write actual code, not just identify yourself as a coder."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])
print("✨ Agent prompts created successfully")

# Create a basic search tool
print("🔍 Initializing Tavily search tool...")
search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
print("✅ Search tool initialized")

# Create the agents
@traceable(name="create_agent")
def create_agent(prompt, agent_type="specialist"):
    """Create an agent with the specified prompt and type."""
    print(f"🤖 Creating {agent_type} agent...")
    chain = prompt | llm
    
    @traceable(name=f"{agent_type}_agent_execution")
    def agent_fn(state):
        print(f"\n📝 {agent_type.title()} agent receiving state: {state}")
        
        # Create a trace config
        config = RunnableConfig(
            tags=[agent_type, "agent_execution"],
            metadata={
                "agent_type": agent_type,
                "timestamp": datetime.datetime.now().isoformat(),
                "input_messages_count": len(state["messages"])
            }
        )
        
        result = chain.invoke(
            {
                "messages": state["messages"],
                "input": state["input"]
            },
            config=config
        )
        
        print(f"💡 {agent_type.title()} agent result: {result}")
        
        # If it's the supervisor, return just the agent name
        if agent_type == "supervisor":
            return {
                "messages": state["messages"] + [HumanMessage(content=result.content.strip("'\""))],
                "next": None,
                "input": state["input"]
            }
        
        # For specialist agents, return their full response
        return {
            "messages": state["messages"] + [HumanMessage(content=result.content)],
            "next": None,
            "input": state["input"]
        }
    
    return agent_fn

print("⚙️ Creating agent chains...")
supervisor_chain = create_agent(supervisor_prompt, "supervisor")
researcher_chain = create_agent(researcher_prompt, "researcher")
writer_chain = create_agent(writer_prompt, "writer")
coder_chain = create_agent(coder_prompt, "coder")
print("✅ Agent chains created successfully")

# Define workflow functions
@traceable(name="route_to_agent")
def route_to_agent(state: AgentState) -> Literal["researcher", "writer", "coder", "end"]:
    """Route the message to the appropriate agent based on supervisor's decision."""
    print("\n🔄 Routing message to appropriate agent...")
    messages = state["messages"]
    last_message = messages[-1].content.lower().strip().replace("'", "").replace('"', "")
    print(f"👨‍💼 Supervisor's decision: {last_message}")
    
    response_map = {
        "researcher": "researcher",
        "research": "researcher",
        "writer": "writer",
        "writing": "writer",
        "write": "writer",
        "coder": "coder",
        "coding": "coder",
        "code": "coder",
        "programmer": "coder"
    }
    
    result = response_map.get(last_message, "end")
    print(f"➡️ Routing to: {result}")
    return result

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determine if the conversation should continue."""
    return "end"

print("🔧 Setting up workflow graph...")
# Create the workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("researcher", researcher_chain)
workflow.add_node("writer", writer_chain)
workflow.add_node("coder", coder_chain)

# Connect supervisor's decisions to specialists
workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "writer": "writer",
        "coder": "coder",
        "end": END
    }
)

# Connect specialists to END
workflow.add_edge("researcher", END)
workflow.add_edge("writer", END)
workflow.add_edge("coder", END)

# Set entry point
workflow.set_entry_point("supervisor")
print("✅ Workflow graph setup complete")

# Compile the workflow
print("⚡ Compiling workflow...")
chain = workflow.compile()
print("✅ Workflow compiled successfully")

# Function to run the multi-agent system
@traceable(name="multi_agent_system")
def run_multi_agent(user_input: str) -> str:
    """Run the multi-agent system with user input."""
    print("\n----------------------------------------")
    print(f"🎯 Processing new user input: {user_input}")
    
    # Create trace config for the full run
    config = RunnableConfig(
        tags=["full_conversation"],
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "input_length": len(user_input)
        }
    )
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "next": None,
        "input": user_input
    }
    
    print("🔄 Running chain with initial state...")
    result = chain.invoke(initial_state, config=config)
    
    final_message = result["messages"][-1].content
    print(f"✨ Final response: {final_message}")
    print("----------------------------------------\n")
    return final_message

# Modify the example usage section
if __name__ == "__main__":
    print("ℹ️ This file is meant to be imported by streamlit_app.py")
    print("🚀 Please run: streamlit run streamlit_app.py")
