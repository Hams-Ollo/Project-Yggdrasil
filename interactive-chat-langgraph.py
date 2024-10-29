import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load environment variables
load_dotenv()

# Initialize Rich console for better formatting
console = Console()

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

# Enhanced system message
sys_msg = SystemMessage(content="""You are a sophisticated AI assistant with access to arithmetic operations and internet search capabilities. Your role is to:

1. Analyze user queries to determine whether they require calculations, information lookup, or both
2. For calculations:
   - Use the provided arithmetic tools (add, multiply, divide) with precision
   - Show your mathematical reasoning step by step
   - Format numerical results clearly

3. For information searches:
   - Use the DuckDuckGo search tool to find current and relevant information
   - Summarize search results concisely and cite sources when possible
   - Verify information accuracy before presenting it

4. When combining calculations and searches:
   - Break down complex problems into clear steps
   - Explain your approach before executing it
   - Present results in a structured, easy-to-follow format

5. Communication style:
   - Be clear, professional, and friendly
   - Use markdown formatting for better readability when appropriate
   - Ask clarifying questions if user input is ambiguous

Remember to use the most appropriate tool for each task and explain your reasoning process.""")

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

def print_message(message):
    """Print a single message with enhanced formatting using Rich."""
    role = "Assistant" if message.type == "ai" else "Human"
    
    if role == "Human":
        console.print(f"\n[bold blue]You:[/bold blue] {message.content}")
    else:
        # Try to parse content as markdown for better formatting
        try:
            markdown = Markdown(message.content)
            console.print(f"\n[bold green]Assistant:[/bold green]")
            console.print(Panel(markdown, border_style="green"))
        except:
            # Fallback to plain text if markdown parsing fails
            console.print(f"\n[bold green]Assistant:[/bold green] {message.content}")

def print_messages(messages):
    """Pretty print all messages with enhanced formatting."""
    console.print("\n" + "="*50, style="bold yellow")
    for msg in messages:
        print_message(msg)
    console.print("="*50 + "\n", style="bold yellow")

def chat_interface():
    """Interactive chat interface for the LangGraph system."""
    console.print("\n[bold cyan]Welcome to the Enhanced AI Assistant Chat![/bold cyan]")
    console.print("""
[yellow]Available capabilities:[/yellow]
• Arithmetic calculations (add, multiply, divide)
• Internet search via DuckDuckGo
• Combined operations (calculations + search)

Type 'exit' to end the conversation.
""")

    # Initialize the graph
    react_graph = build_graph()
    
    while True:
        # Get user input
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[bold red]Chat session terminated.[/bold red]")
            break
            
        # Check for exit command
        if user_input.lower() == 'exit':
            console.print("\n[bold cyan]Thank you for chatting! Goodbye![/bold cyan]")
            break
        
        if not user_input:
            continue
            
        try:
            # Process the input through the graph
            messages = [HumanMessage(content=user_input)]
            result = react_graph.invoke({"messages": messages})
            
            # Display the results
            print_messages(result['messages'])
        except Exception as e:
            console.print(f"\n[bold red]An error occurred: {str(e)}[/bold red]")
            console.print("[yellow]Please try again with a different query.[/yellow]\n")

if __name__ == "__main__":
    chat_interface()
