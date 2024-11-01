#-------------------------------------------------------------------------------------#
# SETUP:
#
# Setup venv and install the requirements
# 1. Create a virtual environment -> python -m venv venv
# 2. Activate the virtual environment -> .\venv\Scripts\Activate
# 3. Install the requirements -> pip install -r requirements.txt
# 4. Run the streamlit app -> streamlit run app.py / streamlit run frontend/streamlit_app.py
# streamlit run chat_app.py
# Git Commands:
# 1. Initialize repository -> git init
# 2. Add files to staging -> git add .
# 3. Commit changes -> git commit -m "your message"
# 4. Create new branch -> git checkout -b branch-name
# 5. Switch branches -> git checkout branch-name
# 6. Push to remote -> git push -u origin branch-name
# 7. Pull latest changes -> git pull origin branch-name
# 8. Check status -> git status
# 9. View commit history -> git log
#-------------------------------------------------------------------------------------#

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolNode, ToolInvocation
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import operator
from typing import Annotated, Sequence, TypedDict, Union
from langchain.tools import StructuredTool
from langsmith import Client
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

# Load environment variables
print("🔑 Loading environment variables...")
load_dotenv()

# Initialize LangSmith client
client = Client()
try:
    # Try to get existing project or create new one
    project = client.create_project("Project-Yggdrasil")
except Exception as e:
    print(f"Note: Project setup error: {e}")
    # Continue anyway as tracing will still work with default project

class ContextualMemory:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the contextual memory system."""
        print("🧠 Initializing contextual memory...")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.memory_index = faiss.IndexFlatL2(self.embedding_dim)
        self.stored_embeddings = []
        self.stored_messages = []
        
    def add_interaction(self, user_message, assistant_message):
        """Store a new interaction with both messages."""
        print("💾 Storing new interaction in memory...")
        context = f"{user_message.content} {assistant_message.content}"
        embedding = self.embedding_model.encode([context])[0].astype('float32')
        
        self.stored_messages.append((user_message, assistant_message))
        self.stored_embeddings.append(embedding)
        
        if len(self.stored_embeddings) == 1:
            self.memory_index.add(embedding.reshape(1, -1))
        else:
            embeddings_array = np.vstack(self.stored_embeddings)
            self.memory_index = faiss.IndexFlatL2(self.embedding_dim)
            self.memory_index.add(embeddings_array)
    
    def get_relevant_context(self, query, k=3):
        """Retrieve relevant context based on the query."""
        print("🔍 Searching memory for relevant context...")
        if not self.stored_embeddings:
            return []
        
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        D, I = self.memory_index.search(query_embedding, min(k, len(self.stored_embeddings)))
        
        relevant_messages = []
        for idx in I[0]:
            if idx < len(self.stored_messages):
                relevant_messages.extend(self.stored_messages[idx])
        
        return relevant_messages

# Initialize LLM
print("🤖 Initializing LLM...")
llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# Base schemas for our arithmetic tools
class ArithmeticInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")

# Multiply Tool
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers together"
    args_schema: Type[BaseModel] = ArithmeticInput
    
    def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> int:
        """Execute the multiplication."""
        print(f"✖️ Multiplying {a} × {b}")
        result = a * b
        print(f"Result: {result}")
        return result
    
    async def _arun(self, a: int, b: int, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> int:
        """Execute multiplication asynchronously."""
        return self._run(a, b, run_manager=run_manager.get_sync() if run_manager else None)

# Add Tool
class AddTool(BaseTool):
    name: str = "add"
    description: str = "Add two numbers together"
    args_schema: Type[BaseModel] = ArithmeticInput
    
    def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> int:
        """Execute the addition."""
        print(f"➕ Adding {a} + {b}")
        result = a + b
        print(f"Result: {result}")
        return result
    
    async def _arun(self, a: int, b: int, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> int:
        """Execute addition asynchronously."""
        return self._run(a, b, run_manager=run_manager.get_sync() if run_manager else None)

# Divide Tool
class DivideTool(BaseTool):
    name: str = "divide"
    description: str = "Divide first number by second number"
    args_schema: Type[BaseModel] = ArithmeticInput
    
    def _run(self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None) -> float:
        """Execute the division."""
        print(f"➗ Dividing {a} ÷ {b}")
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        print(f"Result: {result}")
        return result
    
    async def _arun(self, a: int, b: int, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> float:
        """Execute division asynchronously."""
        return self._run(a, b, run_manager=run_manager.get_sync() if run_manager else None)

# Search Tool
class SearchInput(BaseModel):
    query: str = Field(description="The search query to look up")

class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search the web for information using DuckDuckGo"
    args_schema: Type[BaseModel] = SearchInput
    search_engine: DuckDuckGoSearchRun = Field(default_factory=DuckDuckGoSearchRun)
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute the web search."""
        if not query or not isinstance(query, str):
            raise ValueError(f"Invalid query: {query}")
            
        # Add time context to the query
        current_context = "current 2024"
        enhanced_query = f"{query} {current_context}"
        print(f"🌐 Searching web for: {enhanced_query}")
        
        try:
            result = self.search_engine.run(enhanced_query)
            print("🔎 Search completed")
            
            # Add a note about when this search was performed
            timestamp_note = "\nNote: This search was performed on November 1, 2024."
            return result + timestamp_note
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            raise e
    
    async def _arun(self, query: str, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Execute web search asynchronously."""
        return self._run(query, run_manager=run_manager.get_sync() if run_manager else None)

# Define tools - now just the search tool
tools = [SearchTool()]

# Create tool node
tool_node = ToolNode(tools)

# Update system message to only reference search capability
sys_msg = SystemMessage(content="""You are a sophisticated AI assistant with internet search capabilities. Your role is to:

1. Use the search tool effectively:
   - Use the 'search' tool to find current and relevant information
   - Summarize search results concisely and clearly
   - Verify information accuracy before presenting it
   - Cite sources when appropriate

2. Communication style:
   - Be clear, professional, and friendly
   - Present information in a structured, easy-to-follow format
   - Use markdown formatting for better readability when appropriate

Always use the search tool when looking up information. Stay factual and current.""")

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]

# Define agent functions
def should_use_tool(state: AgentState) -> bool:
    """Decide if we should use the search tool."""
    print("🤔 Deciding if search is needed...")
    
    if not state["messages"]:
        return False
        
    last_message = state["messages"][-1].content.lower()
    
    # Basic greetings and simple responses should not trigger search
    basic_phrases = ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"]
    if any(phrase in last_message for phrase in basic_phrases):
        print("❌ Basic greeting detected, no search needed")
        return False
    
    # Expanded list of search triggers
    search_keywords = [
        "search", "look up", "find", "what is", "tell me about",
        "how to", "where", "when", "who", "why", "latest",
        "news about", "information on", "details about",
        "price", "stock", "market", "current", "recent",
        "update", "status", "info", "can you", "what's the"
    ]
    
    # Check if any keyword is in the message
    needs_search = any(keyword in last_message for keyword in search_keywords)
    print(f"{'✅' if needs_search else '❌'} Search needed: {needs_search}")
    return needs_search

def call_tool(state: AgentState) -> AgentState:
    """Call the search tool."""
    print("🔧 Executing search...")
    last_message = state["messages"][-1].content
    
    try:
        # Create the search tool instance
        search_tool = SearchTool()
        
        # Execute the search directly
        result = search_tool._run(last_message)
        
        # Format the response nicely
        response = (
            "Based on my search, here's what I found:\n\n"
            f"{result}\n\n"
            "Is there anything specific from these results you'd like me to explain further?"
        )
        
        return {"messages": state["messages"] + [AIMessage(content=response)]}
    
    except Exception as e:
        print(f"❌ Tool error: {str(e)}")
        error_msg = (
            "I apologize, but I encountered an error while searching. "
            f"Error details: {str(e)}"
        )
        return {"messages": state["messages"] + [AIMessage(content=error_msg)]}

def agent_response(state: AgentState) -> AgentState:
    """Generate the agent's response."""
    print("💭 Generating response...")
    # Only use the last few messages to prevent context overflow
    recent_messages = state["messages"][-5:]  # Keep only last 5 messages
    messages = [sys_msg] + recent_messages
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}

# Build graph
def build_graph():
    """Build and compile the LangGraph state graph."""
    print("🏗️ Building graph...")
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_response)
    workflow.add_node("tool_caller", call_tool)
    
    # Add edges with modified flow
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_use_tool,
        {
            True: "tool_caller",
            False: END
        }
    )
    # Tool caller should always end after execution
    workflow.add_edge("tool_caller", END)
    
    print("✅ Graph built successfully")
    return workflow.compile()

def print_message(message):
    """Print a single message with clear formatting."""
    role = "🤖 Assistant" if message.type == "ai" else "👤 Human"
    print(f"\n{role}: {message.content}")

def print_messages(messages):
    """Print all messages with clear separation."""
    print("\n" + "="*50)
    for msg in messages:
        print_message(msg)
    print("="*50 + "\n")

def chat_interface():
    """Interactive chat interface with enhanced contextual memory."""
    print("\n👋 Welcome to the Enhanced AI Assistant Chat!")
    print("""
Available capabilities:
• Internet search via DuckDuckGo
• Contextual memory for more coherent conversations

Type 'exit' to end the conversation.
""")

    # Initialize the graph and memory
    react_graph = build_graph()
    memory = ContextualMemory()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == 'exit':
                print("\n👋 Thank you for chatting! Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("🔄 Processing your message...")
            user_message = HumanMessage(content=user_input)
            context_messages = memory.get_relevant_context(user_input)
            messages = context_messages + [user_message]
            
            result = react_graph.invoke({"messages": messages})
            assistant_message = result['messages'][-1]
            
            print_messages([user_message, assistant_message])
            memory.add_interaction(user_message, assistant_message)
            
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("🔄 Please try again with a different query.\n")

def test_search_tool():
    """Test the search tool functionality."""
    search_tool = SearchTool()
    test_query = "What is the latest gold price per ounce real time market data"  # More specific query
    try:
        print(f"Testing search with query: {test_query}")
        result = search_tool._run(test_query)
        print("Search successful!")
        print("Result:", result)
        return True
    except Exception as e:
        print("Search test failed:", str(e))
        return False

if __name__ == "__main__":
    # Test search functionality first
    print("\nTesting search tool...")
    if test_search_tool():
        print("Search tool test passed ✅")
    else:
        print("Search tool test failed ❌")
        
    # Continue with chat interface
    chat_interface()