import streamlit as st
from interactive_chat_langgraph import build_graph, ContextualMemory, sys_msg
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time

def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'graph' not in st.session_state:
        st.session_state.graph = build_graph()
    if 'memory' not in st.session_state:
        st.session_state.memory = ContextualMemory()
        
def display_message(role, content, avatar):
    """Display a single message with a typing effect for AI responses."""
    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            # Simulate typing effect for AI responses
            message_placeholder = st.empty()
            full_response = content
            
            # Split response into smaller chunks for smoother typing effect
            chunk_size = max(len(full_response) // 50, 1)
            for i in range(0, len(full_response) + chunk_size, chunk_size):
                message_placeholder.markdown(full_response[:i] + "▌")
                time.sleep(0.01)
            
            message_placeholder.markdown(full_response)
        else:
            st.markdown(content)

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stMarkdown {
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Header with custom styling
    st.markdown("""
        <h1 style='text-align: center; color: #4a90e2;'>
            🤖 AI Assistant
        </h1>
    """, unsafe_allow_html=True)
    
    # Sidebar with capabilities info
    with st.sidebar:
        st.markdown("### 🛠️ Capabilities")
        
        # Tool sections with expandable details
        with st.expander("🧮 Mathematical Operations"):
            st.markdown("""
                - Addition
                - Multiplication
                - Division
                - Step-by-step solutions
            """)
            
        with st.expander("🌐 Web Search"):
            st.markdown("""
                - Real-time information lookup
                - Source verification
                - Summarized results
            """)
            
        with st.expander("🧠 Contextual Memory"):
            st.markdown("""
                - Remembers conversation history
                - Maintains context
                - Provides relevant references
            """)
        
        st.markdown("### 💡 Tips")
        st.info("""
            - Be specific with your questions
            - For calculations, provide clear numbers
            - For searches, use precise keywords
            - Use natural language
        """)
        
        # Add a clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.memory = ContextualMemory()
            st.experimental_rerun()

    # Main chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            avatar = "🤖" if isinstance(message, AIMessage) else "👤"
            role = "assistant" if isinstance(message, AIMessage) else "user"
            display_message(role, message.content, avatar)

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to state and display
            user_message = HumanMessage(content=prompt)
            st.session_state.messages.append(user_message)
            display_message("user", prompt, "👤")

            try:
                with st.spinner("🤔 Thinking..."):
                    # Get context and process through graph
                    context_messages = st.session_state.memory.get_relevant_context(prompt)
                    messages = context_messages + [user_message]
                    
                    # Process through graph
                    result = st.session_state.graph.invoke({"messages": messages})
                    assistant_message = result['messages'][-1]

                    # Add to memory and state
                    st.session_state.memory.add_interaction(user_message, assistant_message)
                    st.session_state.messages.append(assistant_message)

                    # Display assistant response
                    display_message("assistant", assistant_message.content, "🤖")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.warning("Please try again with a different query.")

if __name__ == "__main__":
    main() 