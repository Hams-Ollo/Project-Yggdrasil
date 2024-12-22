# 🌳 **Project Yggdrasil: The Tree of Knowledge AI Assistant**

**Project Yggdrasil** is an advanced multi-agent AI system designed to act as a personal knowledge repository and management assistant. Inspired by the Norse Tree of Life, Yggdrasil enables users to integrate, organize, and visualize their knowledge, notes, and files into a structured "Tree of Knowledge." Using AI-powered multi-agent orchestration, the system empowers users to streamline knowledge curation and optimize personal or professional knowledge management workflows.

---

## 🎡 **Overview**

**Project Yggdrasil** leverages **LangGraph** for orchestrating intelligent conversations between specialized agents, creating a dynamic knowledge ecosystem. Users can consolidate their notes, files, and data into the system, which uses AI to analyze, connect, and visualize information, enabling effective curation and streamlined access.

### **Core Features**
- **Knowledge Integration**: Seamlessly upload and organize notes, files, and documents.
- **Tree of Knowledge Visualization**: Automatically create a visual representation of your knowledge base to explore connections.
- **Multi-Agent Architecture**:
  - **Supervisor Agent**: Routes user requests to specialized agents.
  - **Researcher Agent**: Handles data gathering and analysis.
  - **Writer Agent**: Manages content creation and editing.
  - **Coder Agent**: Provides programming solutions and explanations.
- **Web Search Integration**: Use external resources for additional context and knowledge.
- **Real-Time Monitoring**: Track interactions and processes in real-time with **LangSmith**.
- **Fallback Intelligence**: Seamlessly switch between Groq LLM and OpenAI APIs for enhanced reliability.

---

## 🛠️ **Technology Stack**
- **LangGraph**: Multi-agent orchestration.
- **Groq LLM**: Primary language model (with OpenAI fallback).
- **Streamlit**: Intuitive web-based user interface.
- **LangSmith**: Monitoring and tracing agent interactions.
- **Tavily**: Web search integration.

---

## 🚀 **Setup Instructions**

### **Prerequisites**
- Python 3.12+
- Virtual Environment
- Required API Keys:
  - Groq API Key
  - OpenAI API Key (fallback)
  - LangSmith API Key
  - Tavily API Key

### **Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/project-yggdrasil.git
   cd project-yggdrasil
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the root directory and add your API keys:
   ```plaintext
   GROQ_API_KEY=your_groq_key
   OPENAI_API_KEY=your_openai_key
   LANGCHAIN_API_KEY=your_langsmith_key
   TAVILY_API_KEY=your_tavily_key
   ```

---

## 🔄 **Running the Application**

1. Start the Streamlit Interface:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Access the web interface at: [http://localhost:8501](http://localhost:8501)

---

## 🔎 **Features**

### **1. Multi-Agent AI Orchestration**
- **Supervisor Agent**: Delegates user requests to the appropriate agents.
- **Researcher Agent**: Performs information gathering and analysis.
- **Writer Agent**: Assists with content creation and editing.
- **Coder Agent**: Handles programming-related queries and code generation.

### **2. Knowledge Base Integration**
- Upload notes, files, and documents to build your personal repository.
- Automatically extract metadata and organize information.
- Create dynamic connections and insights using AI-driven analysis.

### **3. Tree of Knowledge Visualization**
- Generate a visual representation of your knowledge base.
- Explore relationships, patterns, and gaps in your data.

### **4. Real-Time Monitoring and Insights**
- Use **LangSmith** for interaction monitoring and tracing.
- Gain insights into agent performance and task delegation.

### **5. Web Search Integration**
- Use **Tavily** to enhance knowledge retrieval with external sources.

---

## 🕰 **Development Roadmap**

### Upcoming Features
- Add new specialist agents for:
  - **Data Analysis**
  - **Task Planning**
- Implement advanced caching mechanisms for faster retrieval.
- Enhance error handling and recovery workflows.
- Add **conversation memory** and context management.
- Implement user authentication and access controls.
- Support file uploads for batch processing and advanced analysis.

---

## 🤝 **Contributing**

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

---

## 🖋️ **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- [LangChain](https://langchain.com) for providing the **LangGraph** orchestration framework.
- [Groq](https://groq.com) for their cutting-edge LLM API.
- [OpenAI](https://openai.com) for fallback support.
- [Streamlit](https://streamlit.io) for the user-friendly web interface.

---

## 📢 **Support**

For support, please open an issue in the GitHub repository or contact the maintainers.
