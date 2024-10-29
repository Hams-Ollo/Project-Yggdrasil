import os
import sys
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from backend.app.main import run_multi_agent

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
""")

# Custom avatar URLs or Base64 images
USER_AVATAR = "😊"  # Simple smiley face for user
ASSISTANT_AVATAR = "🤖"  # Robot emoji to match app title

# Display chat messages with custom avatars
for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar=USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    ):
        st.markdown(message["content"])

# Chat input with custom avatars
if prompt := st.chat_input("What can I help you with?"):
    # Display user message with custom avatar
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response with custom avatar
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
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
    
    - 🎯 **Supervisor**: Routes your request to the appropriate specialist
    - 📚 Researcher: For finding and analyzing information
    - ✍️ Writer: For writing and editing content
    - 💻 Coder: For programming help and code explanations
    - 💬 General Chat: For casual conversation and general assistance
    
    Each message is analyzed by the supervisor and routed to the most appropriate specialist.
    """)
