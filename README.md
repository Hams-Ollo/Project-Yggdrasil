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

1. Clone the repository
