import streamlit as st
from main import run_multi_agent

# Set page configuration
st.set_page_config(
    page_title="Multi-Agent Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat header
st.title("🤖 Multi-Agent Assistant")
st.markdown("""
This assistant uses multiple AI agents to help you with different tasks:
- 📚 Researcher: For finding and analyzing information
- ✍️ Writer: For writing and editing content
- 💻 Coder: For programming help and code explanations
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What can I help you with?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_multi_agent(prompt)
            st.markdown(response)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a multi-agent AI assistant that uses specialized agents for different tasks:
    
    - **Supervisor**: Routes your request to the appropriate specialist
    - **Researcher**: Handles information gathering and analysis
    - **Writer**: Handles content creation and editing
    - **Coder**: Handles programming-related questions
    
    Each message is analyzed by the supervisor and routed to the most appropriate specialist.
    """)
