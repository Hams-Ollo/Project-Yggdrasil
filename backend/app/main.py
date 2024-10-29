# main.py - Multi-agent chatbot assistant with general conversation capabilities
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
from typing import Annotated, Literal, TypedDict, List, Dict
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
from langsmith import traceable
from langchain_core.tracers import ConsoleCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_core.runnables import RunnableConfig
import datetime

print("📚 All libraries imported successfully")

#-------------------------------------------------------------------------------------#
# LangSmith Setup
#-------------------------------------------------------------------------------------#
print("🔄 Setting up LangSmith tracing...")
try:
    from langsmith import Client
    client = Client()
    
    # Configure LangSmith environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = "Project-Yggdrasil"
    
    callback_manager = CallbackManager([ConsoleCallbackHandler()])
    print("✅ LangSmith configuration loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: LangSmith setup failed: {str(e)}")
    callback_manager = CallbackManager([])

#-------------------------------------------------------------------------------------#
# LLM Setup with Fallback
#-------------------------------------------------------------------------------------#
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
            # Fallback to OpenAI
            from langchain_openai import ChatOpenAI
            fallback_llm = ChatOpenAI(
                temperature=0.1,
                model_name="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY"),
                callbacks=callback_manager.handlers
            )
            print("✅ Fallback to OpenAI successful")
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

#-------------------------------------------------------------------------------------#
# Agent State and Tools Setup
#-------------------------------------------------------------------------------------#
# Define the agent state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str
    input: str

# Initialize search tool
print("🔍 Initializing Tavily search tool...")
search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
print("✅ Search tool initialized")

#-------------------------------------------------------------------------------------#
# Agent Prompts
#-------------------------------------------------------------------------------------#
print("🎯 Setting up agent prompts...")

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent supervisor agent responsible for analyzing user requests and delegating tasks to the most appropriate specialist agent. Your ONLY role is to output the name of the most appropriate agent to handle the request.

    IMPORTANT: You must ONLY respond with one of these exact words: 'researcher', 'writer', 'coder', or 'general'. Do not include any other text, punctuation, or explanation.

    Available Specialists:
    1. RESEARCHER
       - For: Research questions, fact-checking, data analysis, current events
       - Example inputs: "What is quantum computing?", "Tell me about AI trends"
    
    2. WRITER
       - For: Content creation, editing, creative writing
       - Example inputs: "Write a blog post", "Create a story", "Draft an email"
    
    3. CODER
       - For: Programming, debugging, technical implementation
       - Example inputs: "How do I sort in Python?", "Debug this code"

    4. GENERAL
       - For: General conversation, unclear queries, multiple topics, casual chat
       - Example inputs: "How are you?", "What do you think about...", "Can you help me?"
       - Also handles: Ambiguous requests, mixed topics, general advice

    Route to GENERAL if:
    - The request is conversational in nature
    - The category is unclear or spans multiple domains
    - The request doesn't clearly fit other specialists
    - The user is seeking general advice or discussion

    Remember: Respond ONLY with 'researcher', 'writer', 'coder', or 'general'."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research specialist agent with expertise in finding, analyzing, and synthesizing information.
    Your strengths include:
    - Thorough research and fact-checking
    - Data analysis and interpretation
    - Current events and trends analysis
    - Answering complex questions with verified information
    
    Always cite your sources when possible and provide comprehensive, accurate information."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a writing specialist agent with expertise in content creation and editing.
    Your strengths include:
    - Creating engaging and well-structured content
    - Adapting writing style to different purposes
    - Editing and improving existing content
    - Developing creative pieces and professional documents
    
    Focus on clarity, engagement, and meeting the specific writing needs of the user."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a coding specialist agent with expertise in programming and technical implementation.
    Your approach should:
    1. Provide complete, working code solutions
    2. Include clear explanations of how the code works
    3. Follow best practices and patterns
    4. Consider error handling and edge cases
    5. Optimize for readability and maintainability
    
    Always provide fully functional code that can be used directly, with proper imports and setup."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

general_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a general conversation agent skilled in handling diverse queries and engaging in natural dialogue.
    Your capabilities include:
    - Engaging in casual conversation and small talk
    - Providing general advice and guidance
    - Handling multi-topic discussions
    - Clarifying user needs when requests are unclear
    - Offering thoughtful responses to open-ended questions
    
    Approach:
    1. Be conversational and natural in tone
    2. Ask clarifying questions when needed
    3. Provide balanced and thoughtful responses
    4. Acknowledge when a topic might benefit from specialist expertise
    5. Maintain a helpful and engaging dialogue

    If a query would be better handled by a specialist (researcher, writer, or coder), 
    you can mention this while still providing a helpful general response."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

print("✨ Agent prompts created successfully")

#-------------------------------------------------------------------------------------#
# Agent Creation and Routing
#-------------------------------------------------------------------------------------#
@traceable(name="create_agent")
def create_agent(prompt, agent_type="specialist"):
    """Create an agent with the specified prompt and type."""
    print(f"🤖 Creating {agent_type} agent...")
    chain = prompt | llm
    
    @traceable(name=f"{agent_type}_agent_execution")
    def agent_fn(state):
        try:
            print(f"\n📝 {agent_type.title()} agent receiving state")
            
            config = RunnableConfig(
                tags=[agent_type, "agent_execution"],
                metadata={
                    "agent_type": agent_type,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "input_messages_count": len(state["messages"])
                }
            )
            
            # Execute the chain
            result = chain.invoke(
                {
                    "messages": state["messages"],
                    "input": state["input"]
                },
                config=config
            )
            
            # For supervisor, ensure clean output
            if agent_type == "supervisor":
                cleaned_result = result.content.lower().strip().replace("'", "").replace('"', "")
                print(f"🎯 Supervisor decision: {cleaned_result}")
                return {
                    "messages": state["messages"] + [HumanMessage(content=cleaned_result)],
                    "next": None,
                    "input": state["input"]
                }
            
            # For specialist agents
            return {
                "messages": state["messages"] + [HumanMessage(content=result.content)],
                "next": None,
                "input": state["input"]
            }
            
        except Exception as e:
            error_msg = f"Error in {agent_type} agent: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "messages": state["messages"] + [HumanMessage(content=error_msg)],
                "next": None,
                "input": state["input"]
            }
    
    return agent_fn

@traceable(name="route_to_agent")
def route_to_agent(state: AgentState) -> Literal["researcher", "writer", "coder", "general", "end"]:
    """Route the message to the appropriate agent based on supervisor's decision."""
    print("\n🔄 Routing message...")
    
    if not state["messages"]:
        print("⚠️ No messages in state")
        return "end"
    
    # Get the last message from the supervisor
    last_message = state["messages"][-1].content.lower().strip()
    print(f"👨‍💼 Supervisor's raw decision: {last_message}")
    
    # Direct mapping including general agent
    valid_routes = {
        "researcher": "researcher",
        "writer": "writer",
        "coder": "coder",
        "general": "general"
    }
    
    result = valid_routes.get(last_message, "general")  # Default to general instead of end
    print(f"➡️ Routing to: {result}")
    return result

#-------------------------------------------------------------------------------------#
# Workflow Setup
#-------------------------------------------------------------------------------------#
print("⚙️ Creating agent chains...")
supervisor_chain = create_agent(supervisor_prompt, "supervisor")
researcher_chain = create_agent(researcher_prompt, "researcher")
writer_chain = create_agent(writer_prompt, "writer")
coder_chain = create_agent(coder_prompt, "coder")
general_chain = create_agent(general_prompt, "general")
print("✅ Agent chains created successfully")

print("🔧 Setting up workflow graph...")
# Create the workflow graph
workflow = StateGraph(AgentState)

# Add nodes including general agent
workflow.add_node("supervisor", supervisor_chain)
workflow.add_node("researcher", researcher_chain)
workflow.add_node("writer", writer_chain)
workflow.add_node("coder", coder_chain)
workflow.add_node("general", general_chain)

# Connect supervisor's decisions to specialists and general agent
workflow.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "writer": "writer",
        "coder": "coder",
        "general": "general",
        "end": END
    }
)

# Connect all agents to END
workflow.add_edge("researcher", END)
workflow.add_edge("writer", END)
workflow.add_edge("coder", END)
workflow.add_edge("general", END)

# Set entry point
workflow.set_entry_point("supervisor")
print("✅ Workflow graph setup complete")

# Compile the workflow
print("⚡ Compiling workflow...")
chain = workflow.compile()
print("✅ Workflow compiled successfully")

#-------------------------------------------------------------------------------------#
# Multi-Agent System Runner
#-------------------------------------------------------------------------------------#
@traceable(name="multi_agent_system")
def run_multi_agent(user_input: str) -> str:
    """Run the multi-agent system with user input."""
    print("\n----------------------------------------")
    print(f"🎯 New request: {user_input}")
    
    if not user_input.strip():
        return "Error: Empty input received"
    
    config = RunnableConfig(
        tags=["full_conversation"],
        metadata={
            "timestamp": datetime.datetime.now().isoformat(),
            "input_length": len(user_input)
        }
    )
    
    try:
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "next": None,
            "input": user_input
        }
        
        result = chain.invoke(initial_state, config=config)
        
        if not result.get("messages"):
            return "Error: No response generated"
            
        final_message = result["messages"][-1].content
        print(f"✨ Final response: {final_message}")
        print("----------------------------------------\n")
        return final_message
        
    except Exception as e:
        error_msg = f"Error in multi-agent system: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

#-------------------------------------------------------------------------------------#
# Main Entry Point
#-------------------------------------------------------------------------------------#
if __name__ == "__main__":
    print("ℹ️ This file is meant to be imported by streamlit_app.py")
    print("🚀 Please run: streamlit run streamlit_app.py")
    