# Project Yggdrasil - Multi-Agent AI Assistant

![Project Status](https://img.shields.io/badge/status-in_development-yellow)
![Python](https://img.shields.io/badge/python-3.12-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-green)

## Overview

Project Yggdrasil is a multi-agent AI system built using LangGraph for orchestrating conversations between specialized AI agents. The system uses a supervisor agent to delegate tasks to specialist agents based on the user's needs.

## Current Architecture

### Core Components

- **Supervisor Agent**: Routes requests to appropriate specialist agents
- **Researcher Agent**: Handles information gathering and analysis
- **Writer Agent**: Manages content creation and editing
- **Coder Agent**: Provides programming solutions and explanations

### Technology Stack

- LangGraph for agent orchestration
- Groq LLM (with OpenAI fallback)
- Streamlit for web interface
- LangSmith for monitoring and tracing
- Tavily for web search capabilities

## Setup Instructions

### Prerequisites

- Python 3.12+
- Virtual Environment
- Required API Keys:
  - Groq API Key
  - OpenAI API Key (fallback)
  - LangSmith API Key
  - Tavily API Key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/project-yggdrasil.git
   cd project-yggdrasil
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your API keys:

     ```bash
     GROQ_API_KEY=your_groq_key
     OPENAI_API_KEY=your_openai_key
     LANGCHAIN_API_KEY=your_langsmith_key
     TAVILY_API_KEY=your_tavily_key
     ```

### Running the Application

1. Start the Streamlit interface:

   ```bash
   streamlit run streamlit_app.py
   ```

2. Access the web interface at `http://localhost:8501`

## Features

- Multi-agent conversation orchestration
- Automatic task delegation based on request type
- Seamless fallback to alternative LLM providers
- Real-time conversation monitoring
- Web search integration
- Extensible agent architecture

## Development Roadmap

- [ ] Add new specialist agents (Data Analysis, Task Planning)
- [ ] Implement advanced caching mechanisms
- [ ] Enhance error handling and recovery
- [ ] Add conversation memory and context management
- [ ] Implement user authentication
- [ ] Add support for file uploads and processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain team for LangGraph
- Groq for their LLM API
- OpenAI for fallback support
- Streamlit for the web framework

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
