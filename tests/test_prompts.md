# Multi-Agent System Test Prompts

This document contains test prompts organized by agent type, with explanations of what each prompt tests and expected behavior.

## Supervisor Agent Tests

These prompts test the supervisor's ability to route requests to the appropriate specialist.

### Basic Routing Tests

| Prompt | Expected Agent | Purpose |
|--------|---------------|----------|
| "Can you help me with something?" | Supervisor | Tests handling of vague requests |
| "Hello!" | Supervisor | Tests basic greeting handling |
| "What's your name and your functional capabilities?" | Supervisor | Tests basic identity questions |
| "What are you able to specifically assist me with?" | Supervisor | Tests system knowledge |

### Edge Case Routing Tests

| Prompt | Expected Agent | Purpose |
|--------|---------------|----------|
| "Write a python script to calculate the sum of all numbers from 1 to 100 and then explain quantum computing to me." | Supervisor→Coder/Researcher | Tests handling of multi-agent requests |
| "I'm not sure which agent I need" | Supervisor | Tests handling of uncertain requests |
| "Can you switch to a different agent?" | Supervisor | Tests agent switching requests |

## Researcher Agent Tests

These prompts test the researcher's ability to find and analyze information.

### Information Gathering

| Prompt | Purpose |
|--------|----------|
| "What are the latest developments in quantum computing?" | Tests current events knowledge |
| "Explain the difference between AI and machine learning" | Tests technical concept explanation |
| "What are the environmental impacts of electric vehicles?" | Tests comprehensive analysis |

### Analysis & Comparison

| Prompt | Purpose |
|--------|----------|
| "Compare and contrast React and Angular frameworks" | Tests comparative analysis |
| "Analyze the trends in renewable energy adoption" | Tests trend analysis |
| "What are the pros and cons of different Python web frameworks?" | Tests technical evaluation |

### Fact-Checking

| Prompt | Purpose |
|--------|----------|
| "What is the current world population?" | Tests factual data retrieval |
| "What are the most used programming languages in 2024?" | Tests current statistics |
| "What is the market share of different cloud providers?" | Tests market data analysis |

## Writer Agent Tests

These prompts test the writer's content creation and editing abilities.

### Content Creation

| Prompt | Purpose |
|--------|----------|
| "Write a blog post about the future of AI" | Tests long-form content creation |
| "Create a product description for a smart home device" | Tests technical writing |
| "Write a tutorial on getting started with Python" | Tests instructional writing |

### Professional Writing

| Prompt | Purpose |
|--------|----------|
| "Draft an email announcing a new software release" | Tests business writing |
| "Write a professional LinkedIn post about AI trends" | Tests social media content |
| "Create a user guide for a mobile app" | Tests technical documentation |

### Creative Writing

| Prompt | Purpose |
|--------|----------|
| "Write a story about a programmer in the year 2050" | Tests creative storytelling |
| "Create a metaphor explaining how blockchain works" | Tests technical concept simplification |
| "Write a catchy slogan for a tech startup" | Tests concise creative writing |

## Coder Agent Tests

These prompts test the coder's programming and technical abilities.

### Basic Coding Tasks

| Prompt | Purpose |
|--------|----------|
| "Write a Python function to reverse a string" | Tests basic algorithm implementation |
| "Show me how to implement a binary search in JavaScript" | Tests language-specific coding |
| "Create a simple REST API using FastAPI" | Tests framework knowledge |

### Debugging Scenarios

| Prompt | Purpose |
|--------|----------|
| "Debug this code: print(sorted([1,2,3,None,'4'])" | Tests error identification |
| "Why am I getting a KeyError in my Python dictionary?" | Tests error explanation |
| "Fix this JavaScript promise chain" | Tests async code debugging |

### Implementation Requests

| Prompt | Purpose |
|--------|----------|
| "How do I implement authentication in a Flask app?" | Tests security implementation |
| "Create a React component for a todo list" | Tests frontend development |
| "Write a Python script to process CSV files" | Tests file handling |

### Technical Explanations

| Prompt | Purpose |
|--------|----------|
| "Explain how async/await works in Python" | Tests concept explanation |
| "What's the difference between map, filter, and reduce?" | Tests functional programming knowledge |
| "How does garbage collection work in Python?" | Tests system-level understanding |

## Complex Multi-Agent Tests

These prompts test coordination between multiple agents.

### Multi-Step Problems

| Prompt | Expected Flow | Purpose |
|--------|---------------|----------|
| "Research machine learning frameworks, then write code for a simple neural network" | Researcher → Coder | Tests sequential agent handoff |
| "Analyze cloud providers and write a deployment script for AWS" | Researcher → Coder | Tests research-to-implementation |
| "Research microservices architecture and implement a basic service in Python" | Researcher → Coder | Tests architecture knowledge |

### Edge Cases

| Prompt | Expected Flow | Purpose |
|--------|---------------|----------|
| "Can you both write code and analyze its performance?" | Coder → Researcher | Tests cross-agent analysis |
| "I need research on databases and then code to implement one" | Researcher → Coder | Tests knowledge application |
| "Write documentation for this code: def foo(x): return x*2" | Coder → Writer | Tests technical documentation |
