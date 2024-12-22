# 🌳 **Project Y.G.G.D.R.A.S.I.L.**

**Your Generalized Graph-Driven Repository for AI and Scalable Intelligent Learning**

**Project Y.G.G.D.R.A.S.I.L.** is an advanced multi-agent AI system designed to act as a comprehensive knowledge repository and intelligent learning assistant. Inspired by the Norse Tree of Life, **Y.G.G.D.R.A.S.I.L.** enables users to integrate, organize, and visualize their knowledge, notes, and files into a structured "Graph-Driven Tree of Knowledge." Using cutting-edge AI and graph-based orchestration, the system empowers users to streamline knowledge curation, optimize personal or professional workflows, and achieve scalable insights.

---

## 🎡 **Overview**

**Project Y.G.G.D.R.A.S.I.L.** leverages **LangGraph** for orchestrating intelligent conversations between specialized agents, creating a dynamic and interactive knowledge ecosystem. By consolidating diverse information sources, the system uses AI-driven graph technology to connect, analyze, and visualize data relationships, making knowledge exploration and curation seamless and intuitive.

### **Core Features**
- **Graph-Driven Repository**: Visualize relationships between your notes, documents, and data.
- **AI-Driven Knowledge Curation**: Leverage AI to organize and structure your knowledge base.
- **Multi-Agent Architecture**:
  - **Supervisor Agent**: Routes user requests to specialized agents.
  - **Researcher Agent**: Handles data gathering and analysis.
  - **Writer Agent**: Manages content creation and editing.
  - **Coder Agent**: Provides programming solutions and explanations.
- **Scalable Learning**: Tailored insights for personal growth or organizational use cases.
- **Web Search Integration**: Enhance knowledge with external resources for additional context.
- **Real-Time Monitoring**: Track interactions and processes with **LangSmith**.
- **Fallback Intelligence**: Seamlessly switch between **Groq LLM** and **OpenAI APIs** for enhanced reliability.

---

## 🛠️ **Technology Stack**
- **LangGraph**: Multi-agent orchestration for conversation and task delegation.
- **Groq LLM**: Primary language model (with OpenAI fallback).
- **Streamlit**: Intuitive, web-based user interface.
- **LangSmith**: Real-time monitoring and tracing of agent interactions.
- **Tavily**: Web search integration for external knowledge retrieval.

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

### **1. Graph-Driven Repository**
- Create a **Tree of Knowledge** that visualizes the relationships between your documents, notes, and data.
- Explore and interact with knowledge graphs to discover connections, patterns, and gaps in your knowledge base.

### **2. Multi-Agent AI Orchestration**
- **Supervisor Agent**: Delegates user requests to appropriate specialist agents.
- **Researcher Agent**: Performs information gathering and analysis.
- **Writer Agent**: Assists with content creation and editing.
- **Coder Agent**: Handles programming-related queries and solutions.

### **3. Scalable Intelligent Learning**
- Consolidate diverse data sources into a scalable knowledge base.
- Receive AI-driven recommendations for knowledge curation, learning paths, and actionable insights.

### **4. Real-Time Monitoring and Insights**
- Use **LangSmith** for interaction monitoring and tracing.
- Gain insights into agent performance and task delegation.

### **5. Web Search Integration**
- Use **Tavily** to enhance your knowledge retrieval with external resources.

---

## 🕰 **Development Roadmap**

### Upcoming Features
- Add new specialist agents for:
  - **Data Analysis**
  - **Task Planning**
- Implement advanced caching mechanisms for faster data retrieval.
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
