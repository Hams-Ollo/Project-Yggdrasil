import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState, StateGraph, END, START
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import operator
from typing import Annotated, Sequence, TypedDict, Union

# Load environment variables
load_dotenv()

class ContextualMemory:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the contextual memory system."""
        print("Initializing contextual memory...")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.memory_index = faiss.IndexFlatL2(self.embedding_dim)
        self.stored_embeddings = []
        self.stored_messages = []
        
    def add_interaction(self, user_message, assistant_message):
        """Store a new interaction with both messages."""
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
print("Initializing LLM...")
llm = ChatOpenAI(model="gpt-4-0125-preview")

# Define tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    print("Using multiply tool")
    result = a * b
    print(f"Multiplied {a} × {b} = {result}")
    return result

def add(a: int, b: int) -> int:
    """Adds a and b."""
    print("Using add tool")
    result = a + b
    print(f"Added {a} + {b} = {result}")
    return result

def divide(a: int, b: int) -> float:
    """Divide a and b."""
    print("Using divide tool")
    if b == 0:
        raise ValueError("Cannot divide by zero")
    result = a / b
    print(f"Divided {a} ÷ {b} = {result}")
    return result

search = DuckDuckGoSearchRun()
def logged_search(query: str) -> str:
    """Search the web."""
    print(f"Searching for: {query}")
    result = search.invoke(query)
    print("Search completed")
    return result

tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Divide two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "logged_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

# Create tool executor
tool_executor = ToolExecutor({"multiply": multiply, "add": add, "divide": divide, "logged_search": logged_search})

# System message
sys_msg = SystemMessage(content="""You are a sophisticated AI assistant with access to arithmetic operations, internet search capabilities, and contextual memory. Your role is to:

1. Maintain and utilize conversation context:
   - Remember user-provided information
   - Reference previous interactions when relevant
   - Maintain consistency in responses based on known information

2. For calculations:
   - Use the provided arithmetic tools (add, multiply, divide) with precision
   - Show your mathematical reasoning step by step
   - Format numerical results clearly

3. For information searches:
   - Use the logged_search tool to find current and relevant information
   - Summarize search results concisely and cite sources when possible
   - Verify information accuracy before presenting it

4. When combining calculations and searches:
   - Break down complex problems into clear steps
   - Explain your approach before executing it
   - Present results in a structured, easy-to-follow format

5. Communication style:
   - Be clear, professional, and friendly
   - Use markdown formatting for better readability when appropriate
   - Ask clarifying questions only if the information isn't available in the conversation history

Remember to use the most appropriate tool for each task and maintain continuity across the conversation.""")

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]

# Define agent functions
def should_use_tool(state: AgentState) -> bool:
    """Decide if we should use a tool."""
    messages = [sys_msg] + state["messages"]
    response = llm.invoke(messages)
    
    tool_calls = getattr(response, "tool_calls", None)
    return bool(tool_calls)

def call_tool(state: AgentState) -> AgentState:
    """Call the appropriate tool."""
    messages = [sys_msg] + state["messages"]
    response = llm.invoke(messages)
    
    tool_calls = response.tool_calls
    if not tool_calls:
        return state
    
    tool_results = []
    for tool_call in tool_calls:
        action = ToolInvocation(
            tool=tool_call.function.name,
            tool_input=tool_call.function.arguments
        )
        result = tool_executor.invoke(action)
        tool_results.append(result)
    
    new_messages = state["messages"] + [AIMessage(content=str(tool_results))]
    return {"messages": new_messages}

def agent_response(state: AgentState) -> AgentState:
    """Generate the agent's response."""
    messages = [sys_msg] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}

# Build graph
def build_graph():
    """Build and compile the LangGraph state graph."""
    print("Building graph...")
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_response)
    workflow.add_node("tool_caller", call_tool)
    
    # Add edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_use_tool,
        {
            True: "tool_caller",
            False: END
        }
    )
    workflow.add_edge("tool_caller", "agent")
    
    print("Graph built successfully")
    return workflow.compile()

def print_message(message):
    """Print a single message with clear formatting."""
    role = "Assistant" if message.type == "ai" else "Human"
    print(f"\n{role}: {message.content}")

def print_messages(messages):
    """Print all messages with clear separation."""
    print("\n" + "="*50)
    for msg in messages:
        print_message(msg)
    print("="*50 + "\n")

def chat_interface():
    """Interactive chat interface with enhanced contextual memory."""
    print("\nWelcome to the Enhanced AI Assistant Chat!")
    print("""
Available capabilities:
• Arithmetic calculations (add, multiply, divide)
• Internet search via DuckDuckGo
• Combined operations (calculations + search)
• Contextual memory for more coherent conversations

Type 'exit' to end the conversation.
""")

    # Initialize the graph and memory
    react_graph = build_graph()
    memory = ContextualMemory()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nThank you for chatting! Goodbye!")
                break
                
            if not user_input:
                continue
                
            user_message = HumanMessage(content=user_input)
            context_messages = memory.get_relevant_context(user_input)
            messages = context_messages + [user_message]
            
            result = react_graph.invoke({"messages": messages})
            assistant_message = result['messages'][-1]
            
            print_messages([user_message, assistant_message])
            memory.add_interaction(user_message, assistant_message)
            
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different query.\n")

if __name__ == "__main__":
    chat_interface()