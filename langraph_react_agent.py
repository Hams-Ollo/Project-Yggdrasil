import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Define tool functions
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

# Initialize search tool
search = DuckDuckGoSearchRun()

# Combine all tools
tools = [add, multiply, divide, search]
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs.")

# Node for reasoning
def reasoner(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("reasoner", reasoner)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "reasoner")
    builder.add_conditional_edges(
        "reasoner",
        tools_condition,
    )
    builder.add_edge("tools", "reasoner")
    return builder.compile()

def print_messages(messages):
    """Pretty print messages with clear formatting."""
    print("\n" + "="*50)
    for msg in messages:
        role = "Assistant" if msg.type == "ai" else "Human"
        print(f"\n{role}: {msg.content}")
    print("\n" + "="*50 + "\n")

def chat_interface():
    """Interactive chat interface for the LangGraph system."""
    print("\nWelcome to the AI Assistant Chat!")
    print("You can ask questions, perform calculations, or search for information.")
    print("Type 'exit' to end the conversation.\n")

    # Initialize the graph
    react_graph = build_graph()
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit command
        if user_input.lower() == 'exit':
            print("\nThank you for chatting! Goodbye!")
            break
        
        # Process the input through the graph
        messages = [HumanMessage(content=user_input)]
        result = react_graph.invoke({"messages": messages})
        
        # Display the results
        print_messages(result['messages'])

if __name__ == "__main__":
    chat_interface()