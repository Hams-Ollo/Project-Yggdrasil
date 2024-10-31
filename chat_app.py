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

import streamlit as st
from interactive_chat_langgraph import build_graph, ContextualMemory, sys_msg
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time

# Initialize session state
def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ContextualMemory()
    if 'graph' not in st.session_state:
        st.session_state.graph = build_graph()

def display_message(message, is_user: bool):
    """Display a single message with appropriate styling"""
    with st.chat_message("😊" if is_user else "🤖"):
        st.markdown(message.content)

def display_chat_history():
    """Display all messages in the chat history"""
    for message in st.session_state.messages:
        is_user = isinstance(message, HumanMessage)
        display_message(message, is_user)

def process_user_input(user_input: str) -> None:
    """Process user input and generate response"""
    if user_input:
        # Create user message
        user_message = HumanMessage(content=user_input)
        st.session_state.messages.append(user_message)
        
        # Get relevant context from memory
        context_messages = st.session_state.memory.get_relevant_context(user_input)
        messages = context_messages + [user_message]
        
        # Generate response
        with st.spinner("Thinking..."):
            result = st.session_state.graph.invoke({"messages": messages})
            assistant_message = result['messages'][-1]
            
        # Store in memory and session state
        st.session_state.memory.add_interaction(user_message, assistant_message)
        st.session_state.messages.append(assistant_message)

def create_sidebar():
    """Create the sidebar with information and controls"""
    with st.sidebar:
        st.title("🤖 AI Assistant")
        st.markdown("""
        ### Capabilities:
        - ✨ Contextual Memory
        - 🧮 Arithmetic Operations
        - 🔍 Web Search
        - 🤝 Combined Operations
        
        ### Available Tools:
        1. `multiply`: Multiply two numbers
        2. `add`: Add two numbers
        3. `divide`: Divide two numbers
        4. `logged_search`: Search the web
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.memory = ContextualMemory()
            st.rerun()

def main():
    # Set page configuration
    st.set_page_config(
        page_title="AI Assistant Chat",
        page_icon="🤖",
        layout="wide",
    )
    
    # Initialize session state
    init_session_state()
    
    # Create sidebar
    create_sidebar()
    
    # Main chat container
    st.title("💬 Chat with AI Assistant")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        process_user_input(user_input)
        st.rerun()

if __name__ == "__main__":
    main()