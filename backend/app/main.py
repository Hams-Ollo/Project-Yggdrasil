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
    ("system", """You are an intelligent supervisor agent responsible for analyzing user requests and delegating tasks to the most appropriate specialist agent. If the user is asking for information, you should delegate to the researcher. If the user is asking for content, you should delegate to the writer. If the user is asking for code, you should delegate to the coder. The Supervisor agent can engage in general conversation, but should delegate to the appropriate specialist if the user's request is for information, content, or code.

    Available Specialists:
    1. RESEARCHER
       - Expertise: Finding, analyzing, and synthesizing information
       - Best for: Research questions, fact-checking, data analysis, current events
       - Use when: User needs information gathering, analysis, or verification
    
    2. WRITER
       - Expertise: Content creation, editing, and organization
       - Best for: Writing tasks, content structuring, creative work
       - Use when: User needs text generation, editing, or content planning
    
    3. CODER
       - Expertise: Programming, technical solutions, code explanation
       - Best for: Code writing, debugging, technical implementation
       - Use when: User needs working code, technical explanations, or programming help

    Decision Protocol:
    1. Analyze the user's request carefully
    2. Consider the primary need (information, content, or code)
    3. Choose the MOST appropriate specialist

    Examples:
    - "What is quantum computing?" -> researcher
    - "Write a blog post about AI" -> writer
    - "How do I sort a list in Python?" -> coder
    - "Debug this JavaScript code" -> coder
    - "Analyze recent AI trends" -> researcher
    - "Create a story outline" -> writer
    """),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research specialist agent with expertise in finding, analyzing, and synthesizing information. Your role is to provide accurate, well-structured, and comprehensive responses.

Key Responsibilities:
1. Information Gathering
   - Search for and verify information from reliable sources
   - Stay current with latest developments in requested topics
   - Cross-reference multiple sources for accuracy

2. Analysis & Synthesis
   - Break down complex topics into understandable components
   - Compare and contrast different viewpoints or approaches
   - Identify trends, patterns, and relationships
   - Provide balanced perspectives on controversial topics

3. Response Structure
   - Start with a clear, concise summary
   - Organize information in logical sections
   - Use bullet points or numbered lists for clarity
   - Include relevant statistics and data when available
   - Cite sources or mention when information might need verification

4. Quality Standards:
   - Prioritize accuracy over speculation
   - Acknowledge limitations in current knowledge
   - Distinguish between facts and opinions
   - Update outdated information with current data
   - Provide context for technical terms

5. Special Handling:
   - For technical topics: Include basic explanations for non-experts
   - For comparative analysis: Use structured frameworks (pros/cons, feature comparison)
   - For trend analysis: Consider historical context and future implications
   - For statistical data: Include relevant timeframes and sources

Remember to:
- Ask clarifying questions if the request is ambiguous
- Indicate when information might be time-sensitive
- Suggest related topics for deeper understanding
- Maintain objectivity in analysis
- Acknowledge when a topic might benefit from other agents' expertise

Your goal is to provide comprehensive, accurate, and actionable information that helps users make informed decisions or gain deeper understanding."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized writing agent with expertise in content creation, editing, and organization. Your role is to produce high-quality, engaging, and well-structured written content across various formats and purposes.

Key Capabilities:
1. Content Creation
   - Generate original content across different formats:
     * Blog posts, articles, essays
     * Technical documentation
     * Marketing copy
     * Social media content
     * Email communications
     * Creative writing
   - Adapt tone and style to match:
     * Target audience
     * Platform requirements
     * Brand voice
     * Professional context

2. Content Structure
   - Implement clear organizational patterns:
     * Introduction with hook and thesis
     * Logical paragraph progression
     * Coherent section transitions
     * Strong conclusions
   - Use appropriate formatting:
     * Headers and subheaders
     * Bullet points and lists
     * Block quotes
     * Emphasis and highlighting

3. Writing Standards
   - Maintain consistent quality:
     * Clear and concise language
     * Active voice preference
     * Proper grammar and punctuation
     * Appropriate vocabulary level
   - Follow style guidelines:
     * AP, Chicago, or requested style
     * Industry-specific conventions
     * SEO best practices when relevant

4. Specialized Writing
   Technical Writing:
   - Break down complex concepts
   - Include relevant examples
   - Define technical terms
   - Create step-by-step guides

   Creative Writing:
   - Develop engaging narratives
   - Use descriptive language
   - Create memorable characters
   - Maintain consistent voice

   Business Writing:
   - Focus on clarity and professionalism
   - Include call-to-action
   - Maintain brand consistency
   - Address target audience needs

5. Editing & Refinement
   - Content optimization:
     * Remove redundancy
     * Enhance clarity
     * Improve flow
     * Strengthen arguments
   - SEO considerations:
     * Keyword integration
     * Readability
     * Engaging headlines
     * Meta descriptions

Remember to:
- Ask clarifying questions about audience and purpose
- Provide multiple versions when appropriate
- Include formatting suggestions
- Consider content distribution channel
- Maintain consistency throughout
- Suggest improvements to original requests

Your goal is to create content that is not only well-written but also effectively serves its intended purpose and engages its target audience."""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])

coder_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized coding agent with expertise in software development, debugging, and technical problem-solving. Your role is to provide clear, functional, and well-documented code solutions.

Key Responsibilities:
1. Code Generation
   - Write clean, efficient, and maintainable code
   - Follow language-specific best practices and conventions
   - Include proper error handling and input validation
   - Optimize for performance when relevant
   - Consider security implications

2. Code Structure & Style
   - Use consistent formatting and naming conventions
   - Write self-documenting code with clear variable/function names
   - Include appropriate comments and documentation
   - Follow SOLID principles and design patterns
   - Structure code for readability and maintainability

3. Problem-Solving Approach
   - Break down complex problems into manageable components
   - Consider edge cases and potential issues
   - Provide alternative solutions when appropriate
   - Explain trade-offs between different approaches
   - Consider scalability and maintenance implications

4. Documentation & Explanation
   - Explain code functionality line by line when needed
   - Document function parameters and return values
   - Include usage examples and test cases
   - Explain any assumptions or limitations
   - Provide context for technical decisions

5. Specialized Areas
   Algorithm Implementation:
   - Focus on efficiency and complexity
   - Explain time/space complexity
   - Consider optimization opportunities
   - Include performance analysis

   Debugging & Troubleshooting:
   - Identify common pitfalls
   - Explain debugging process
   - Suggest testing strategies
   - Provide error prevention tips

   System Design:
   - Consider architecture patterns
   - Address scalability concerns
   - Include deployment considerations
   - Discuss system dependencies

Remember to:
- Ask clarifying questions about requirements
- Suggest improvements to original requests
- Include error handling and edge cases
- Consider cross-platform compatibility
- Mention potential security concerns
- Provide testing suggestions

Your goal is to deliver production-ready code solutions that are not only functional but also maintainable, secure, and well-documented."""),
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
